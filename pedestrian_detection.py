import cv2
import numpy as np
import time
from typing import List
from ultralytics import YOLO
from datatypes import PedestrianRecord
from utils import rect_in_zone


class PedestrianDetector:
    def __init__(
            self,
            model_path="yolov8n.pt",
            alert_time=3.0,
            overlap_thresh=0.25,
            conf_thresh=0.4,
            miss_limit=10,
            frame_skip=2,
            color_outside=(0, 255, 0),
            color_entered=(0, 165, 255),
            color_alert=(0, 0, 255),
            color_zone=(255, 0, 0)
    ):
        self.alert_time = alert_time
        self.overlap_thresh = overlap_thresh
        self.conf_thresh = conf_thresh
        self.miss_limit = miss_limit
        self.frame_skip = frame_skip

        self.color_outside = color_outside
        self.color_entered = color_entered
        self.color_alert = color_alert
        self.color_zone = color_zone

        self.model = YOLO(model_path)

    def process_video(
            self,
            video_source: str,
            polygon_pts: list,
            window_name: str = "Pedestrian Detector"
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

        track_state = {}
        last_results = []
        completed_records: List[PedestrianRecord] = []

        print(f"Press 'q' or 'ESC' to stop.")

        while cap.isOpened():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

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
                    active_ids.add(tid)
                    in_zone = rect_in_zone(x1, y1, x2, y2, zone_mask) >= self.overlap_thresh

                    if in_zone:
                        if tid not in track_state:
                            track_state[tid] = {
                                "entry_ts": timestamp,
                                "missed": 0,
                                "qualified": False
                            }

                        track_state[tid]["missed"] = 0
                        dwell = timestamp - track_state[tid]["entry_ts"]

                        # Pedestrian is qualified if he waits 3 seconds
                        if dwell >= self.alert_time:
                            track_state[tid]["qualified"] = True

                        color = self.color_alert if track_state[tid]["qualified"] else self.color_entered
                    else:
                        # Left the zone
                        if tid in track_state:
                            state = track_state[tid]
                            if state["qualified"]:
                                record = PedestrianRecord(
                                    id=int(tid),
                                    type="person",
                                    appeared_at=round(state["entry_ts"], 2),
                                    left=round(timestamp, 2),
                                    waiting_time=round(timestamp - state["entry_ts"], 2)
                                )
                                completed_records.append(record)
                            del track_state[tid]

                        color = self.color_outside

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, color, -1)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Clean up lost IDs
            for tid in list(track_state.keys()):
                if tid not in active_ids:
                    track_state[tid]["missed"] += 1
                    if track_state[tid]["missed"] > self.miss_limit:
                        state = track_state[tid]
                        if state["qualified"]:
                            record = PedestrianRecord(
                                id=int(tid),
                                type="person",
                                appeared_at=round(state["entry_ts"], 2),
                                left=round(timestamp, 2),
                                waiting_time=round(timestamp - state["entry_ts"], 2)
                            )
                            completed_records.append(record)
                        del track_state[tid]

            # Drawing ROI overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_pts], (255, 50, 50))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [polygon_pts], isClosed=True, color=self.color_zone, thickness=2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            elapsed = time.time() - frame_start
            remaining = frame_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # Handle pedestrians still in zone when video ends
        for tid, state in track_state.items():
            if state["qualified"]:
                record = PedestrianRecord(
                    id=int(tid),
                    type="person",
                    appeared_at=round(state["entry_ts"], 2),
                    left=round(timestamp, 2),
                    waiting_time=round(timestamp - state["entry_ts"], 2)
                )
                completed_records.append(record)

        cap.release()
        cv2.destroyAllWindows()

        return completed_records
