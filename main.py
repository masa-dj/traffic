import cv2
import numpy as np
import time
from ultralytics import YOLO

VIDEO_PATH = "Import.mp4"
MODEL_PATH = "yolov8n.pt"

ALERT_TIME = 3
OVERLAP_THRESH = 0.25
CONF_THRESH = 0.4
MISS_LIMIT = 10
FRAME_SKIP = 2

POLYGON_PTS = np.array([
    [360, 380],
    [750, 360],
    [770, 455],
    [350, 475]
], dtype=np.int32)

COLOR_OUTSIDE = (0, 255, 0)
COLOR_ENTERED = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_ZONE = (255, 0, 0)


def rect_in_zone(x1, y1, x2, y2, mask):
    h, w = mask.shape
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, w), min(y2, h)

    if x2c <= x1c or y2c <= y1c:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return 0.0

    return np.count_nonzero(mask[y1c:y2c, x1c:x2c]) / box_area


def main():
    print(f"Loading YOLO model ({MODEL_PATH})...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_PATH}'. Check file path.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_duration = 1.0 / fps

    zone_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [POLYGON_PTS], 255)

    track_state = {}
    last_results = []

    print("Press 'q' or 'ESC' on the video window to exit.")

    while cap.isOpened():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Video stream finished or disconnected.")
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_idx / fps

        # YOLO detection
        if frame_idx % FRAME_SKIP == 0:
            last_results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=CONF_THRESH,
                verbose=False,
                tracker="bytetrack.yaml"
            )
        results = last_results

        alert_triggered = False
        active_ids = set()

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                active_ids.add(tid)
                in_zone = rect_in_zone(x1, y1, x2, y2, zone_mask) >= OVERLAP_THRESH

                if in_zone:
                    if tid not in track_state:
                        track_state[tid] = {"entry_ts": timestamp, "missed": 0}
                    track_state[tid]["missed"] = 0
                    dwell = timestamp - track_state[tid]["entry_ts"]

                    color = COLOR_ALERT if dwell >= ALERT_TIME else COLOR_ENTERED
                    if dwell >= ALERT_TIME:
                        alert_triggered = True
                else:
                    if tid in track_state:
                        del track_state[tid]
                    color = COLOR_OUTSIDE

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, color, -1)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for tid in list(track_state.keys()):
            if tid not in active_ids:
                track_state[tid]["missed"] += 1
                if track_state[tid]["missed"] > MISS_LIMIT:
                    del track_state[tid]

        overlay = frame.copy()
        cv2.fillPoly(overlay, [POLYGON_PTS], (255, 50, 50))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [POLYGON_PTS], isClosed=True, color=COLOR_ZONE, thickness=2)

        if alert_triggered:
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 200), -1)
            cv2.putText(frame, f"ALERT: Person in zone  [{timestamp:.1f}s]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            print(f"[ALERT] Person in zone at {timestamp:.2f}s")

        cv2.putText(frame,
                    f"FPS: {fps:.0f}  |  Frame: {frame_idx}  |  Tracked: {len(track_state)}",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show video
        cv2.imshow("Pedestrian Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        elapsed = time.time() - frame_start
        remaining = frame_duration - elapsed
        if remaining > 0:
            time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()