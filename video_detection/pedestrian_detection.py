import threading
from typing import Callable, List, Optional, Set

import numpy as np

from datatypes import PedestrianRecord
from video_detection.base_detector import BaseDetector, Detection
from video_detection.utils import rect_in_zone


class PedestrianDetector(BaseDetector):
    tracker = "bytetrack.yaml"
    overlay_color = (255, 50, 50)
    marker_radius = 5

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
        super().__init__(model_path, conf_thresh, frame_skip,
                         target_classes=[0], color_zone=color_zone)
        self.alert_time = alert_time
        self.overlap_thresh = overlap_thresh
        self.leave_grace = leave_grace
        self.filter_zone_origin = filter_zone_origin
        self.origin_bottom_frac = origin_bottom_frac
        self.filter_moving_up = filter_moving_up
        self.move_up_thresh = move_up_thresh

        self.color_outside = color_outside
        self.color_entered = color_entered
        self.color_alert = color_alert
        self.color_excluded = color_excluded

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
        return self.run(video_source, polygon_pts, window_name,
                        on_qualified, stop_event, show_window, on_frame)

    def _reset_state(self) -> None:
        self.track_state = {}
        self.notified_ids = set()
        self.seen_ids = set()
        self.excluded_ids = set()
        self.first_cy = {}
        self.completed_records: List[PedestrianRecord] = []

        zone = self.zone
        self.origin_min_cy = zone.max_y - (zone.max_y - zone.min_y) * self.origin_bottom_frac

    def _update_track(self, det: Detection, timestamp: float):
        in_zone = self._in_zone(det)
        self._register_first_sight(det, in_zone)
        self._exclude_if_moving_up(det)

        if det.tid in self.excluded_ids:
            return self.color_excluded
        if in_zone:
            return self._track_in_zone(det.tid, timestamp)
        return self._track_outside(det.tid, timestamp)

    def _in_zone(self, det: Detection) -> bool:
        return rect_in_zone(*det.box, self.zone.mask) >= self.overlap_thresh

    def _register_first_sight(self, det: Detection, in_zone: bool) -> None:
        if det.tid in self.seen_ids:
            return
        cy = det.center[1]
        self.seen_ids.add(det.tid)
        self.first_cy[det.tid] = cy
        if self.filter_zone_origin and in_zone and cy >= self.origin_min_cy:
            self.excluded_ids.add(det.tid)

    def _exclude_if_moving_up(self, det: Detection) -> None:
        if det.tid in self.excluded_ids or not self.filter_moving_up:
            return
        if self.first_cy[det.tid] - det.center[1] > self.move_up_thresh:
            self.excluded_ids.add(det.tid)
            self.track_state.pop(det.tid, None)

    def _track_in_zone(self, tid: int, timestamp: float):
        state = self.track_state.setdefault(
            tid, {"entry_ts": timestamp, "last_in_zone": timestamp, "qualified": False})
        state["last_in_zone"] = timestamp
        self._qualify_if_waited(tid, state, timestamp)
        return self._state_color(state)

    def _qualify_if_waited(self, tid: int, state: dict, timestamp: float) -> None:
        dwell = timestamp - state["entry_ts"]
        if dwell < self.alert_time or state["qualified"]:
            return
        state["qualified"] = True
        if tid not in self.notified_ids:
            self.notified_ids.add(tid)
            self._notify(tid, timestamp, dwell)

    def _track_outside(self, tid: int, timestamp: float):
        state = self.track_state.get(tid)
        if state is None:
            return self.color_outside
        if self._left_zone(state, timestamp):
            self._close_track(tid)
            return self.color_outside
        return self._state_color(state)

    def _state_color(self, state: dict):
        return self.color_alert if state["qualified"] else self.color_entered

    def _left_zone(self, state: dict, timestamp: float) -> bool:
        return timestamp - state["last_in_zone"] > self.leave_grace

    def _close_track(self, tid: int) -> None:
        state = self.track_state.pop(tid)
        if state["qualified"]:
            self.completed_records.append(self._make_record(tid, state))

    def _handle_lost(self, active_ids: Set[int], timestamp: float) -> None:
        for tid in list(self.track_state):
            if tid not in active_ids and self._left_zone(self.track_state[tid], timestamp):
                self._close_track(tid)

    def _collect_results(self) -> List[PedestrianRecord]:
        for tid in list(self.track_state):
            self._close_track(tid)
        return self.completed_records

    @staticmethod
    def _make_record(tid, state) -> PedestrianRecord:
        return PedestrianRecord(
            id=int(tid),
            type="person",
            appeared_at=round(state["entry_ts"], 2),
            left=round(state["last_in_zone"], 2),
            waiting_time=round(state["last_in_zone"] - state["entry_ts"], 2)
        )

    def _label(self, det: Detection) -> str:
        return f"ID {det.tid}"
