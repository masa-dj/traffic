import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class Detection(NamedTuple):
    tid: int
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    cls_id: int

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


@dataclass
class Zone:
    polygon: np.ndarray
    mask: np.ndarray
    min_y: int
    max_y: int

    @classmethod
    def from_points(cls, points, width: int, height: int) -> "Zone":
        polygon = np.array(points, dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return cls(polygon, mask, int(polygon[:, 1].min()), int(polygon[:, 1].max()))


class BaseDetector(ABC):

    tracker: str = "bytetrack.yaml"
    overlay_color: Tuple[int, int, int] = (255, 255, 255)
    marker_radius: int = 4

    def __init__(self, model_path: str, conf_thresh: float, frame_skip: int,
                 target_classes: List[int], color_zone: Tuple[int, int, int]):
        self.conf_thresh = conf_thresh
        self.frame_skip = frame_skip
        self.target_classes = target_classes
        self.color_zone = color_zone
        self.model = YOLO(model_path)

        self.zone: Optional[Zone] = None
        self._on_event: Optional[Callable[..., None]] = None
        self._last_results: list = []

    @abstractmethod
    def _reset_state(self) -> None:
        """Initialise per-run tracking state (self.zone is already set)."""

    @abstractmethod
    def _update_track(self, det: Detection, timestamp: float) -> Tuple[int, int, int]:
        """Update state for one detection and return its drawing colour."""

    @abstractmethod
    def _handle_lost(self, active_ids: Set[int], timestamp: float) -> None:
        """Clean up tracks that were not seen in this frame."""

    @abstractmethod
    def _collect_results(self) -> List[Any]:
        """Return the final results once the video has ended."""

    @abstractmethod
    def _label(self, det: Detection) -> str:
        """Text drawn above the bounding box."""

    def _on_new_frame(self, timestamp: float) -> None:
        """Optional per-frame hook, called before detection."""

    def run(
            self,
            video_source: str,
            zone_pts: List[List[int]],
            window_name: str,
            on_event: Optional[Callable[..., None]] = None,
            stop_event: Optional[threading.Event] = None,
            show_window: bool = True,
            on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> List[Any]:
        cap = self._open_capture(video_source)
        if cap is None:
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._start_session(cap, zone_pts, on_event, show_window)

        while cap.isOpened() and not self._stopped(stop_event):
            frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx, timestamp = self._frame_clock(cap, fps)
            self._on_new_frame(timestamp)

            detections = self._detect(frame, frame_idx)
            active_ids = self._process_detections(frame, detections, timestamp)
            self._handle_lost(active_ids, timestamp)

            self._draw_zone(frame, show_window, on_frame)
            if on_frame is not None:
                on_frame(frame)
            if self._show_and_check_quit(frame, window_name, show_window):
                break

            self._pace(frame_start, fps)

        self._release(cap, window_name, show_window)
        return self._collect_results()

    @staticmethod
    def _open_capture(video_source: str) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{video_source}'.")
            return None
        return cap

    def _start_session(self, cap, zone_pts, on_event, show_window: bool) -> None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.zone = Zone.from_points(zone_pts, width, height)
        self._on_event = on_event
        self._last_results = []
        self._reset_state()
        if show_window:
            print("Press 'q' or 'ESC' to stop.")

    @staticmethod
    def _release(cap, window_name: str, show_window: bool) -> None:
        cap.release()
        if show_window:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    @staticmethod
    def _stopped(stop_event: Optional[threading.Event]) -> bool:
        return stop_event is not None and stop_event.is_set()

    @staticmethod
    def _frame_clock(cap, fps: float) -> Tuple[int, float]:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        return frame_idx, frame_idx / fps

    def _detect(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        if frame_idx % self.frame_skip == 0 or not self._last_results:
            self._last_results = self.model.track(
                frame,
                persist=True,
                classes=self.target_classes,
                conf=self.conf_thresh,
                verbose=False,
                tracker=self.tracker,
            )
        return self._extract_detections(self._last_results)

    @staticmethod
    def _extract_detections(results) -> List[Detection]:
        detections = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for box, tid, cls_id in zip(boxes, ids, clss):
                detections.append(Detection(int(tid), tuple(int(v) for v in box), int(cls_id)))
        return detections

    def _process_detections(self, frame, detections: List[Detection], timestamp: float) -> Set[int]:
        active_ids = set()
        for det in detections:
            active_ids.add(det.tid)
            color = self._update_track(det, timestamp)
            self._draw_detection(frame, det, color)
        return active_ids

    def _draw_detection(self, frame, det: Detection, color) -> None:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, det.center, self.marker_radius, color, -1)
        cv2.putText(frame, self._label(det), (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_zone(self, frame, show_window: bool, on_frame) -> None:
        if not (show_window or on_frame is not None):
            return
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone.polygon], self.overlay_color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [self.zone.polygon], isClosed=True,
                      color=self.color_zone, thickness=2)

    @staticmethod
    def _show_and_check_quit(frame, window_name: str, show_window: bool) -> bool:
        if not show_window:
            return False
        cv2.imshow(window_name, frame)
        return (cv2.waitKey(1) & 0xFF) in (ord('q'), 27)

    @staticmethod
    def _pace(frame_start: float, fps: float) -> None:
        remaining = 1.0 / fps - (time.time() - frame_start)
        if remaining > 0:
            time.sleep(remaining)

    def _notify(self, *args) -> None:
        if self._on_event is not None:
            self._on_event(*args)
