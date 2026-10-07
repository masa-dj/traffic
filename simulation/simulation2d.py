import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from intersection import Intersection
from controllers import FixedCycleController, SmartCycleController
from video_detection.pedestrian_detection import PedestrianDetector
from video_detection.vehicle_detection import VehicleDetector

TICK_MS = 1000
# how often the UI checks for detector events
POLL_MS = 30

PED_MODEL_PATH = "../yolo26s.pt"
VEHICLE_MODEL_PATH = "../yolo26s.pt"

# detector preview width in pixels
PREVIEW_WIDTH = 640

PED_ZONES = {
    "up": [[170, 230], [380, 225], [410, 330], [150, 350]],
    "down": [[235, 200], [410, 195], [460, 270], [210, 275]],
}

VEHICLE_ZONES = {
    "right": [[230, 20], [410, 20], [550, 380], [150, 400],],
    "left": [[300, 50], [350, 50], [520, 450],[150, 450],],
}


# Match the simulation seconds and button callbacks so they can function on same clock
class SimClock:
    def __init__(self):
        self.t = 0


# prepare frames and send them to ui thread
def push_frame(frames, key, frame):
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH:
        frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    q = frames[key]
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(rgb)
    except queue.Full:
        pass


# Thread to run pedestrian detection
def run_ped_detection(direction, video_path, events, frames, stop_event):
    key = ("pedestrian", direction)

    def on_qualified(tid, ts, dwell):
        events.put(("arrival", *key))

    def on_frame(frame):
        push_frame(frames, key, frame)
    try:
        detector = PedestrianDetector(model_path=PED_MODEL_PATH)
        detector.process_video(
            video_source=video_path,
            polygon_pts=PED_ZONES[direction],
            on_qualified=on_qualified,
            stop_event=stop_event,
            show_window=False,  # the main thread crashes on macOS
            on_frame=on_frame,
        )
    except Exception as exc:
        events.put(("error", *key, str(exc)))
    finally:
        events.put(("done", *key))


# Thread to run vehicle detection
def run_vehicle_detection(direction, video_path, events, frames, stop_event):
    key = ("vehicle", direction)

    def on_arrival(tid, v_type, ts):
        events.put(("arrival", *key))

    def on_frame(frame):
        push_frame(frames, key, frame)
    try:
        detector = VehicleDetector(model_path=VEHICLE_MODEL_PATH)
        detector.process_video(
            video_source=video_path,
            zone_pts=VEHICLE_ZONES[direction],
            on_arrival=on_arrival,
            stop_event=stop_event,
            show_window=False,
            on_frame=on_frame,
        )
    except Exception as exc:
        events.put(("error", *key, str(exc)))
    finally:
        events.put(("done", *key))


DETECTION_TARGETS = {
    "pedestrian": run_ped_detection,
    "vehicle": run_vehicle_detection,
}


def main():
    root = tk.Tk()
    root.title("Intersection Simulation")

    smart = Intersection(root, SmartCycleController(), title="Smart cycle")
    fixed = Intersection(root, FixedCycleController(), title="Fixed cycle")

    smart.grid(row=0, column=0, padx=10, pady=10)
    fixed.grid(row=0, column=1, padx=10, pady=10)

    clock = SimClock()
    events = queue.Queue()  # worker -> UI

    all_keys = [("pedestrian", d) for d in PED_ZONES] + [("vehicle", d) for d in VEHICLE_ZONES]
    frames = {key: queue.Queue(maxsize=1) for key in all_keys}  # latest frame per (kind, direction)
    workers = {}
    buttons = {}

    def tick():
        smart.update(clock.t)
        fixed.update(clock.t)
        clock.t += 1
        root.after(TICK_MS, tick)

    def send_arrival(kind, direction):
        arrival = {
            "type": kind,
            "direction": direction,
            "arrival_time": clock.t,
        }
        smart.add_arrival(arrival)
        fixed.add_arrival(arrival)

    def start_detection(kind, direction):
        key = (kind, direction)
        path = filedialog.askopenfilename(
            title=f"Choose video for {kind}s ({direction})",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        # Drop any leftover frame from a previous run
        try:
            frames[key].get_nowait()
        except queue.Empty:
            pass

        stop_event = threading.Event()

        # Detection window
        window = tk.Toplevel(root)
        window.title(f"{kind.capitalize()} Detector - {direction}")
        label = tk.Label(window)
        label.pack()
        window.protocol("WM_DELETE_WINDOW", stop_event.set)

        workers[key] = {
            "stop": stop_event,
            "window": window,
            "label": label,
            "photo": None,
        }
        buttons[key].config(state="disabled")  # one run per (kind, direction)

        # creating thread
        threading.Thread(
            target=DETECTION_TARGETS[kind],
            args=(direction, path, events, frames, stop_event),
            daemon=True,
        ).start()

    # Tkinter thread: detector events, show frames
    def poll():
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break

            etype = event[0]
            kind = event[1]
            direction = event[2]
            if etype == "arrival":
                send_arrival(kind, direction)
            elif etype == "error":
                message = event[3]
                messagebox.showerror("Detection error", f"{kind} ({direction}): {message}")
            elif etype == "done":
                key = (kind, direction)
                worker = workers.pop(key, None)
                if worker:
                    worker["window"].destroy()
                buttons[key].config(state="normal")

        # show frames
        for key, worker in list(workers.items()):
            try:
                rgb = frames[key].get_nowait()
            except queue.Empty:
                continue
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            worker["label"].config(image=photo)
            worker["photo"] = photo  # keep a reference or tkinter drops the image

        root.after(POLL_MS, poll)

    def on_close():
        for worker in workers.values():
            worker["stop"].set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # One button per (kind, direction)
    button_specs = (
        ("pedestrian", "up"),
        ("pedestrian", "down"),
        ("vehicle", "right"),
        ("vehicle", "left"),
    )
    for row, (kind, direction) in enumerate(button_specs, start=1):
        btn = tk.Button(
            root,
            text=f"Detect {kind}s ({direction})",
            command=lambda k=kind, d=direction: start_detection(k, d),
        )
        btn.grid(row=row, column=0, columnspan=2, pady=(10 if row == 1 else 5, 0))
        buttons[(kind, direction)] = btn

    tick()
    poll()
    root.mainloop()


if __name__ == "__main__":
    main()