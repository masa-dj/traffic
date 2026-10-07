import random
import tkinter as tk
from PIL import Image, ImageTk

CAR_COLOR = {
    "green": "green",
    "yellow": "yellow",
    "red": "red",
    "red+yellow": "orange",  # ask at work
}
PED_COLOR = {
    "green": "green",
    "red": "red",
}

# Colors picked at random for each new marker.
PEDESTRIAN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
VEHICLE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#808000",
]

# Marker sizes
PEDESTRIAN_RADIUS = 6
VEHICLE_WIDTH = 60
VEHICLE_HEIGHT = 40

PEDESTRIAN_OFFSET = 10   # pedestrians spread sideways
VEHICLE_OFFSET = 65


# Draw lights over intersection image
class Intersection:
    IMAGE_PATH = "simulation/intersection.png"
    SCALE = 0.5

    def __init__(self, parent, controller, title=None):
        self.controller = controller

        self.frame = tk.Frame(parent)
        if title:
            tk.Label(self.frame, text=title).pack()

        self.bg_image_pil = Image.open(self.IMAGE_PATH)
        width = int(self.bg_image_pil.width * self.SCALE)
        height = int(self.bg_image_pil.height * self.SCALE)
        resized_image = self.bg_image_pil.resize((width, height), Image.Resampling.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(resized_image)

        self.width = self.bg_image.width()
        self.height = self.bg_image.height()

        self.canvas = tk.Canvas(self.frame, width=self.width, height=self.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.bg_image, anchor=tk.NW)

        # Coordinates for triangles
        self.CAR_right_1_coord = [325, 75, 400, 95, 370, 130]
        self.CAR_right_2_coord = [280, 510, 355, 500, 325, 465]
        self.CAR_left_1_coord = [235, 95, 310, 75, 265, 130]
        self.CAR_left_2_coord = [200, 510, 275, 510, 230, 465]
        self.PED_up_coord = [320, 85, 335, 115, 305, 115]
        self.PED_down_coord = [277, 500, 292, 470, 262, 470]

        # Starting coordinates for pedestrians and cars
        self.PED_start = {"up": (315, 125), "down": (277, 455)}
        self.CAR_start = {"right": (470, 200), "left": (170, 340)}

        self.light_1a = self.canvas.create_polygon(self.CAR_right_1_coord, fill="red", outline="black", width=2)
        self.light_2a = self.canvas.create_polygon(self.CAR_right_2_coord, fill="red", outline="black", width=2)
        self.light_1b = self.canvas.create_polygon(self.CAR_left_1_coord, fill="red", outline="black", width=2)
        self.light_2b = self.canvas.create_polygon(self.CAR_left_2_coord, fill="red", outline="black", width=2)
        self.light_3 = self.canvas.create_polygon(self.PED_up_coord, fill="red", outline="black", width=2)
        self.light_4 = self.canvas.create_polygon(self.PED_down_coord, fill="red", outline="black", width=2)

        self.car_lights = (self.light_1a, self.light_1b, self.light_2a, self.light_2b)
        self.ped_lights = (self.light_3, self.light_4)

        self.arrivals = []
        self.departures = []

    def add_arrival(self, arrival):
        arrival = dict(arrival)
        kind = arrival["type"]
        direction = arrival["direction"]

        waiting = []
        for a in self.arrivals:
            if a["type"] == kind and a["direction"] == direction:
                waiting.append(a)
        index = len(waiting)

        x, y = self._marker_point(kind, direction, index)

        if kind == "pedestrian":
            r = PEDESTRIAN_RADIUS
            color = random.choice(PEDESTRIAN_COLORS)
            marker = self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill=color, outline="black"
            )
        elif kind == "vehicle":
            w, h = VEHICLE_WIDTH / 2, VEHICLE_HEIGHT / 2
            color = random.choice(VEHICLE_COLORS)
            marker = self.canvas.create_rectangle(
                x - w, y - h, x + w, y + h, fill=color, outline="black"
            )
        else:
            raise ValueError(f"Unknown arrival type: {kind!r}")

        arrival["marker"] = marker
        self.arrivals.append(arrival)

        # notify smart cycle that there is a participant waiting
        if hasattr(self.controller, "notify_arrival"):
            self.controller.notify_arrival(kind)
        return arrival

    def _marker_point(self, kind, direction, index=0):
        if kind == "pedestrian":
            x, y = self.PED_start[direction]
            return x + index * PEDESTRIAN_OFFSET, y

        elif kind == "vehicle":
            x, y = self.CAR_start[direction]
            if direction == "right":
                return x + index * VEHICLE_OFFSET, y
            else:
                return x - index * VEHICLE_OFFSET, y

        raise ValueError(f"Unknown arrival type: {kind!r}")

    def _release(self, t, kind):
        still_waiting = []
        for arrival in self.arrivals:
            if arrival["type"] == kind:
                self.canvas.delete(arrival["marker"])
                arrival["wait_time"] = t - arrival["arrival_time"]
                self.departures.append(arrival)
            else:
                still_waiting.append(arrival)
        self.arrivals = still_waiting

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def update(self, t):
        car_lamp, ped_lamp = self.controller.state_at(t)
        for light in self.car_lights:
            self.canvas.itemconfig(light, fill=CAR_COLOR[car_lamp])
        for light in self.ped_lights:
            self.canvas.itemconfig(light, fill=PED_COLOR[ped_lamp])

        if ped_lamp == "green":
            self._release(t, "pedestrian")
        if car_lamp == "green":
            self._release(t, "vehicle")
