import threading
import cv2
import numpy as np
import time
from typing import Callable, Optional, List, Dict, Any
from ultralytics import YOLO


class VehicleDetector:
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
        self.conf_thresh = conf_thresh
        self.miss_limit = miss_limit
        self.frame_skip = frame_skip
        self.relink_window = relink_window
        self.relink_max_dist = relink_max_dist

        # COCO Vehicle Classes
        # 1 - bikes, 2 - cars, 3 - buses, 5 - trucks
        # 0 - people because yolo has problem with motorbikes
        self.target_classes = target_classes or [0, 1, 2, 3, 5, 7]

        self.color_outside = color_outside
        self.color_inside = color_inside
        self.color_zone = color_zone

        self.model = YOLO(model_path)

    # remove "false" new IDs
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

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{video_source}'.")
            return []

        zone_polygon = np.array(zone_pts, dtype=np.int32)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps

        track_state: Dict[int, Dict[str, Any]] = {}
        notified_ids = set()
        departures: Dict[int, Dict[str, Any]] = {}
        last_results = []

        arrivals_log: List[Dict[str, Any]] = []

        if show_window:
            print(f"Press 'q' or 'ESC' to stop.")

        while cap.isOpened():
            if stop_event is not None and stop_event.is_set():
                break

            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = frame_idx / fps

            # remove departures older than the relink
            for d_tid in [d for d, dep in departures.items()
                          if timestamp - dep["last_ts"] > self.relink_window]:
                del departures[d_tid]

            # Detection: uses BotSort
            if frame_idx % self.frame_skip == 0 or not last_results:
                last_results = self.model.track(
                    frame,
                    persist=True,
                    classes=self.target_classes,
                    conf=self.conf_thresh,
                    verbose=False,
                    tracker="botsort.yaml"
                )
            results = last_results

            active_ids = set()

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), tid, cls_id in zip(boxes, ids, clss):
                    tid = int(tid)
                    active_ids.add(tid)

                    # Taking the center of vehicle
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    v_type = self.model.names.get(cls_id, "vehicle")

                    # Is center of box inside of ROI
                    is_inside = cv2.pointPolygonTest(zone_polygon, (float(cx), float(cy)), False) >= 0

                    if is_inside:
                        if tid not in track_state:
                            match_tid = self._find_relink(
                                departures, cx, cy, timestamp,
                                self.relink_window, self.relink_max_dist
                            )
                            if match_tid is not None:
                                # same vehicle continuing under a new tracker ID
                                # remove it
                                del departures[match_tid]
                                notified_ids.add(tid)

                            track_state[tid] = {
                                "type": v_type, "missed": 0,
                                "last_cx": cx, "last_cy": cy, "last_ts": timestamp,
                            }
                        else:
                            track_state[tid]["missed"] = 0
                            track_state[tid]["last_cx"] = cx
                            track_state[tid]["last_cy"] = cy
                            track_state[tid]["last_ts"] = timestamp

                        if tid not in notified_ids:
                            notified_ids.add(tid)
                            entry_time = round(timestamp, 2)
                            arrivals_log.append({"type": v_type, "arrival": entry_time})
                            if on_arrival is not None:
                                on_arrival(tid, v_type, entry_time)

                        color = self.color_inside
                    else:
                        # remember where car left ROI, so a
                        # tracker ID switch after can be relinked
                        if tid in track_state:
                            state = track_state[tid]
                            departures[tid] = {
                                "last_cx": state["last_cx"],
                                "last_cy": state["last_cy"],
                                "last_ts": timestamp,
                            }
                            del track_state[tid]

                        color = self.color_outside

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    cv2.putText(frame, f"{v_type.upper()} {tid}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Lost objects
            for tid in list(track_state.keys()):
                if tid not in active_ids:
                    track_state[tid]["missed"] += 1
                    if track_state[tid]["missed"] > self.miss_limit:
                        state = track_state[tid]
                        departures[tid] = {
                            "last_cx": state["last_cx"],
                            "last_cy": state["last_cy"],
                            "last_ts": state["last_ts"],
                        }
                        del track_state[tid]

            if show_window or on_frame is not None:
                # Drawing the ROI
                overlay = frame.copy()
                cv2.fillPoly(overlay, [zone_polygon], (0, 255, 255))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                cv2.polylines(frame, [zone_polygon], isClosed=True, color=self.color_zone, thickness=2)

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

        cap.release()
        if show_window:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

        return arrivals_log
