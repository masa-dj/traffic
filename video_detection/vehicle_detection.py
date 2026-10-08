import threading
from typing import Any, Callable, Dict, List, Optional, Set

import cv2
import numpy as np

from video_detection.base_detector import BaseDetector, Detection


class VehicleDetector(BaseDetector):
    tracker = "botsort.yaml"
    overlay_color = (0, 255, 255)
    marker_radius = 4

    def __init__(
            self,
            model_path="yolov8s.pt",
            conf_thresh=0.25,
            miss_limit=10,
            frame_skip=2,
            relink_window=2.0,
            relink_max_dist=80.0,
            target_classes: Optional[List[int]] = None,
            color_outside=(0, 255, 0),
            color_inside=(0, 0, 255),
            color_zone=(255, 255, 0)
    ):
        # COCO classes: 1 - bikes, 2 - cars, 3 - motorbikes, 5 - buses, 7 - trucks
        # 0 - people, because yolo has problems with motorbikes
        super().__init__(model_path, conf_thresh, frame_skip,
                         target_classes=target_classes or [0, 1, 2, 3, 5, 7],
                         color_zone=color_zone)
        self.miss_limit = miss_limit
        self.relink_window = relink_window
        self.relink_max_dist = relink_max_dist

        self.color_outside = color_outside
        self.color_inside = color_inside

    def process_video(
            self,
            video_source: str,
            zone_pts: List[List[int]],
            window_name: str = "Vehicle Detector",
            on_arrival: Optional[Callable[[int, str, float], None]] = None,
            stop_event: Optional[threading.Event] = None,
            show_window: bool = True,
            on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> List[Dict[str, Any]]:
        return self.run(video_source, zone_pts, window_name,
                        on_arrival, stop_event, show_window, on_frame)

    def _reset_state(self) -> None:
        self.track_state: Dict[int, Dict[str, Any]] = {}
        self.departures: Dict[int, Dict[str, Any]] = {}
        self.notified_ids = set()
        self.arrivals_log: List[Dict[str, Any]] = []

    def _on_new_frame(self, timestamp: float) -> None:
        """Forget departures that are too old to be relinked."""
        expired = [tid for tid, dep in self.departures.items()
                   if timestamp - dep["last_ts"] > self.relink_window]
        for tid in expired:
            del self.departures[tid]

    def _update_track(self, det: Detection, timestamp: float):
        v_type = self._vehicle_type(det)
        if self._is_inside(det):
            self._on_inside(det, v_type, timestamp)
            return self.color_inside
        self._on_outside(det.tid, timestamp)
        return self.color_outside

    def _vehicle_type(self, det: Detection) -> str:
        return self.model.names.get(det.cls_id, "vehicle")

    def _is_inside(self, det: Detection) -> bool:
        cx, cy = det.center
        return cv2.pointPolygonTest(self.zone.polygon, (float(cx), float(cy)), False) >= 0

    def _on_inside(self, det: Detection, v_type: str, timestamp: float) -> None:
        if det.tid not in self.track_state:
            self._start_track(det, v_type, timestamp)
        else:
            self._refresh_track(det, timestamp)
        self._log_arrival_once(det.tid, v_type, timestamp)

    def _start_track(self, det: Detection, v_type: str, timestamp: float) -> None:
        cx, cy = det.center
        match_tid = self._find_relink(self.departures, cx, cy, timestamp,
                                      self.relink_window, self.relink_max_dist)
        if match_tid is not None:
            # same vehicle continuing under a new tracker ID: not a new arrival
            del self.departures[match_tid]
            self.notified_ids.add(det.tid)
        self.track_state[det.tid] = {
            "type": v_type, "missed": 0,
            "last_cx": cx, "last_cy": cy, "last_ts": timestamp,
        }

    def _refresh_track(self, det: Detection, timestamp: float) -> None:
        cx, cy = det.center
        self.track_state[det.tid].update(
            missed=0, last_cx=cx, last_cy=cy, last_ts=timestamp)

    def _log_arrival_once(self, tid: int, v_type: str, timestamp: float) -> None:
        if tid in self.notified_ids:
            return
        self.notified_ids.add(tid)
        entry_time = round(timestamp, 2)
        self.arrivals_log.append({"type": v_type, "arrival": entry_time})
        self._notify(tid, v_type, entry_time)

    def _on_outside(self, tid: int, timestamp: float) -> None:
        """Remember where the vehicle left the ROI so an ID switch can be relinked."""
        state = self.track_state.pop(tid, None)
        if state is not None:
            self.departures[tid] = {
                "last_cx": state["last_cx"], "last_cy": state["last_cy"],
                "last_ts": timestamp,
            }

    @staticmethod
    def _find_relink(departures, cx, cy, timestamp, relink_window, relink_max_dist):
        best_tid, best_dist = None, None
        for d_tid, dep in departures.items():
            if timestamp - dep["last_ts"] > relink_window:
                continue
            dist = ((cx - dep["last_cx"]) ** 2 + (cy - dep["last_cy"]) ** 2) ** 0.5
            if dist <= relink_max_dist and (best_dist is None or dist < best_dist):
                best_tid, best_dist = d_tid, dist
        return best_tid

    def _handle_lost(self, active_ids: Set[int], timestamp: float) -> None:
        for tid in list(self.track_state):
            if tid in active_ids:
                continue
            self.track_state[tid]["missed"] += 1
            if self.track_state[tid]["missed"] > self.miss_limit:
                self._move_to_departures(tid)

    def _move_to_departures(self, tid: int) -> None:
        state = self.track_state.pop(tid)
        self.departures[tid] = {
            "last_cx": state["last_cx"], "last_cy": state["last_cy"],
            "last_ts": state["last_ts"],
        }

    def _collect_results(self) -> List[Dict[str, Any]]:
        return self.arrivals_log

    def _label(self, det: Detection) -> str:
        return f"{self._vehicle_type(det).upper()} {det.tid}"