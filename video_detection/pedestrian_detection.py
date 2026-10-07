import threading
import cv2
import numpy as np
import time
from typing import Callable, List, Optional
from ultralytics import YOLO
from datatypes import PedestrianRecord
from video_detection.utils import rect_in_zone


class PedestrianDetector:
    def __init__(
            self,
            model_path="yolov8n.pt",
            alert_time=3.0,
            overlap_thresh=0.25,
            conf_thresh=0.4,
            leave_grace=2.0,
            frame_skip=2,
            filter_zone_origin=True,
            origin_bottom_frac=0.5,
            filter_moving_up=True,
            move_up_thresh=60.0,
            color_outside=(0, 255, 0),
            color_entered=(0, 165, 255),
            color_alert=(0, 0, 255),
            color_excluded=(128, 128, 128),
            color_zone=(255, 0, 0)
    ):
        self.alert_time = alert_time
        self.overlap_thresh = overlap_thresh
        self.conf_thresh = conf_thresh
        self.leave_grace = leave_grace
        self.frame_skip = frame_skip
        self.filter_zone_origin = filter_zone_origin
        self.origin_bottom_frac = origin_bottom_frac
        self.filter_moving_up = filter_moving_up
        self.move_up_thresh = move_up_thresh

        self.color_outside = color_outside
        self.color_entered = color_entered
        self.color_alert = color_alert
        self.color_excluded = color_excluded
        self.color_zone = color_zone

        self.model = YOLO(model_path)

    # Create proper type from available data
    @staticmethod
    def _make_record(tid, state) -> PedestrianRecord:
        return PedestrianRecord(
            id=int(tid),
            type="person",
            appeared_at=round(state["entry_ts"], 2),
            left=round(state["last_in_zone"], 2),
            waiting_time=round(state["last_in_zone"] - state["entry_ts"], 2)
        )

    def process_video(
            self,
            video_source: str,
            polygon_pts: list,
            window_name: str = "Pedestrian Detector",
            on_qualified: Optional[Callable[[int, float, float], None]] = None,
            stop_event: Optional[threading.Event] = None,
            show_window: bool = True,
            on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> List[PedestrianRecord]:

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{video_source}'.")
            return []

        polygon_pts = np.array(polygon_pts, dtype=np.int32)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_duration = 1.0 / fps

        zone_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [polygon_pts], 255)

        # in order to filter out people who come from above, find y coordinates
        zone_min_y = int(polygon_pts[:, 1].min())
        zone_max_y = int(polygon_pts[:, 1].max())
        origin_min_cy = zone_max_y - (zone_max_y - zone_min_y) * self.origin_bottom_frac

        track_state = {}
        # remove duplicates
        notified_ids = set()
        seen_ids = set()
        excluded_ids = set()
        first_cy = {}
        last_results = []

        completed_records: List[PedestrianRecord] = []
        timestamp = 0.0

        if show_window:
            print(f"Press 'q' or 'ESC' to stop.")

        while cap.isOpened():
            # threading support
            if stop_event is not None and stop_event.is_set():
                break

            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # detection and tracking
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = frame_idx / fps

            if frame_idx % self.frame_skip == 0 or not last_results:
                last_results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=self.conf_thresh,
                    verbose=False,
                    tracker="bytetrack.yaml"
                )
            results = last_results

            active_ids = set()

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), tid in zip(boxes, ids):
                    tid = int(tid)
                    active_ids.add(tid)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    in_zone = rect_in_zone(x1, y1, x2, y2, zone_mask) >= self.overlap_thresh

                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        first_cy[tid] = cy
                        if self.filter_zone_origin and in_zone and cy >= origin_min_cy:
                            excluded_ids.add(tid)

                    if tid not in excluded_ids and self.filter_moving_up:
                        drifted_up = first_cy[tid] - cy
                        if drifted_up > self.move_up_thresh:
                            excluded_ids.add(tid)
                            track_state.pop(tid, None)

                    if tid in excluded_ids:

                        color = self.color_excluded

                    elif in_zone:
                        if tid not in track_state:
                            track_state[tid] = {
                                "entry_ts": timestamp,
                                "last_in_zone": timestamp,
                                "qualified": False
                            }

                        state = track_state[tid]
                        state["last_in_zone"] = timestamp
                        dwell = timestamp - state["entry_ts"]

                        # Pedestrian is qualified if he waits 3 seconds
                        if dwell >= self.alert_time and not state["qualified"]:
                            state["qualified"] = True
                            if tid not in notified_ids:
                                notified_ids.add(tid)
                                if on_qualified is not None:
                                    on_qualified(tid, timestamp, dwell)

                        color = self.color_alert if state["qualified"] else self.color_entered
                    else:
                        state = track_state.get(tid)
                        if state is None:
                            color = self.color_outside
                        elif timestamp - state["last_in_zone"] > self.leave_grace:
                            if state["qualified"]:
                                completed_records.append(self._make_record(tid, state))
                            del track_state[tid]
                            color = self.color_outside
                        else:
                            color = self.color_alert if state["qualified"] else self.color_entered

                    # draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # cleanup of disappearing pedestrians
            for tid in list(track_state.keys()):
                if tid not in active_ids:
                    state = track_state[tid]
                    if timestamp - state["last_in_zone"] > self.leave_grace:
                        if state["qualified"]:
                            completed_records.append(self._make_record(tid, state))
                        del track_state[tid]

            if show_window or on_frame is not None:
                # Drawing ROI overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_pts], (255, 50, 50))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                cv2.polylines(frame, [polygon_pts], isClosed=True, color=self.color_zone, thickness=2)

            if on_frame is not None:
                on_frame(frame)

            if show_window:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

            elapsed = time.time() - frame_start
            remaining = frame_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

        for tid, state in track_state.items():
            if state["qualified"]:
                completed_records.append(self._make_record(tid, state))

        cap.release()
        if show_window:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

        return completed_records
