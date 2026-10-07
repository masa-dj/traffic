import tkinter as tk
from PIL import Image, ImageTk

CAR_COLOR = {
    "green": "green",
    "yellow": "yellow",
    "red": "red",
    "red+yellow": "orange",
}
PED_COLOR = {
    "green": "green",
    "red": "red",
}

CAR_HEADWAY = 2

SIDES = {
    ("pedestrian", "up"): ((315, 125), "Pešaci ↑"),
    ("pedestrian", "down"): ((277, 455), "Pešaci ↓"),
    ("vehicle", "right"): ((470, 200), "Vozila →"),
    ("vehicle", "left"): ((170, 340), "Vozila ←"),
}
KIND_NAME = {"pedestrian": "Pešaci", "vehicle": "Vozila"}


class Intersection:
    IMAGE_PATH="intersection.png"
    SCALE = 0.5

    def __init__(self, parent, controller, title=None):
        self.controller = controller

        self.frame = tk.Frame(parent)
        if title:
            tk.Label(self.frame, text=title, font=("Arial", 14, "bold")).pack()

        pil = Image.open(self.IMAGE_PATH)
        w = int(pil.width * self.SCALE)
        h = int(pil.height * self.SCALE)
        self.bg_image = ImageTk.PhotoImage(pil.resize((w, h), Image.Resampling.LANCZOS))

        self.canvas = tk.Canvas(self.frame, width=self.bg_image.width(), height=self.bg_image.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.bg_image, anchor=tk.NW)

        car_1a = [325, 75, 400, 95, 370, 130]
        car_2a = [280, 510, 355, 500, 325, 465]
        car_1b = [235, 95, 310, 75, 265, 130]
        car_2b = [200, 510, 275, 510, 230, 465]
        ped_up = [320, 85, 335, 115, 305, 115]
        ped_down = [277, 500, 292, 470, 262, 470]

        def tri(coords):
            return self.canvas.create_polygon(coords, fill="red", outline="black", width=2)

        self.car_lights = tuple(tri(c) for c in (car_1a, car_1b, car_2a, car_2b))
        self.ped_lights = tuple(tri(c) for c in (ped_up, ped_down))

        self.counter_labels = {}
        for key, ((x, y), name) in SIDES.items():
            lbl = tk.Label(self.canvas, text=f"{name}: 0", bg="white", relief="solid", bd=1,
                           font=("Arial", 11, "bold"))
            self.canvas.create_window(x, y, window=lbl)
            self.counter_labels[key] = lbl

        self.stats_label = tk.Label(self.frame, justify="left", font=("Arial", 11))
        self.stats_label.pack(pady=5)

        self.waiting = {key: [] for key in SIDES}
        self.pending = {key: [] for key in SIDES}
        self.wait_sum = {"pedestrian": 0.0, "vehicle": 0.0}
        self.wait_n = {"pedestrian": 0, "vehicle": 0}
        self.car_lamp = "red"
        self.ped_lamp = "red"
        self.now = 0
        self._refresh()

    def add_arrival(self, arrival):
        kind = arrival["type"]
        key = (kind, arrival["direction"])
        if key not in SIDES:
            raise ValueError(f"Unknown arrival: {key!r}")

        lamp = self.ped_lamp if kind == "pedestrian" else self.car_lamp
        if lamp == "green":
            self._record(kind, 0.0)
        else:
            self.waiting[key].append(arrival["arrival_time"])

        if hasattr(self.controller, "notify_arrival"):
            self.controller.notify_arrival(kind)
        self._refresh()

    def _record(self, kind, wait):
        self.wait_sum[kind] += wait
        self.wait_n[kind] += 1

    def _release(self, t, kind):
        step = CAR_HEADWAY if kind == "vehicle" else 0
        for key in SIDES:
            if key[0] != kind:
                continue
            for k, arrived in enumerate(sorted(self.waiting[key])):
                delay = k * step
                self._record(kind, t - arrived + delay)
                self.pending[key].append(t + delay)
            self.waiting[key] = []

    def _refresh(self):
        t = self.now
        for key, (_, name) in SIDES.items():
            self.pending[key] = [d for d in self.pending[key] if d > t]
            count = len(self.waiting[key]) + len(self.pending[key])
            self.counter_labels[key].config(text=f"{name}: {count}")

        lines = ["Prosečno čekanje:"]
        for kind in ("pedestrian", "vehicle"):
            n = self.wait_n[kind]
            avg = self.wait_sum[kind] / n if n else 0.0
            lines.append(f"  {KIND_NAME[kind]}: {avg:.1f} s  (opsluženo: {n})")
        self.stats_label.config(text="\n".join(lines))

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def update(self, t):
        self.now = t
        car_lamp, ped_lamp = self.controller.state_at(t)
        self.car_lamp, self.ped_lamp = car_lamp, ped_lamp

        for light in self.car_lights:
            self.canvas.itemconfig(light, fill=CAR_COLOR[car_lamp])
        for light in self.ped_lights:
            self.canvas.itemconfig(light, fill=PED_COLOR[ped_lamp])

        if ped_lamp == "green":
            self._release(t, "pedestrian")
        if car_lamp == "green":
            self._release(t, "vehicle")
        self._refresh()