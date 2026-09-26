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

        self.light_1a = self.canvas.create_polygon(self.CAR_right_1_coord, fill="red", outline="black", width=2)
        self.light_2a = self.canvas.create_polygon(self.CAR_right_2_coord, fill="red", outline="black", width=2)
        self.light_1b = self.canvas.create_polygon(self.CAR_left_1_coord, fill="red", outline="black", width=2)
        self.light_2b = self.canvas.create_polygon(self.CAR_left_2_coord, fill="red", outline="black", width=2)
        self.light_3 = self.canvas.create_polygon(self.PED_up_coord, fill="red", outline="black", width=2)
        self.light_4 = self.canvas.create_polygon(self.PED_down_coord, fill="red", outline="black", width=2)

        self.car_lights = (self.light_1a, self.light_1b, self.light_2a, self.light_2b)
        self.ped_lights = (self.light_3, self.light_4)

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


if __name__ == "__main__":
    from controllers import FixedCycleController

    root = tk.Tk()
    root.title("Intersection")
    ix = Intersection(root, FixedCycleController())
    ix.pack()

    t = 0

    def tick():
        global t
        ix.update(t)
        t += 1
        root.after(1000, tick)

    tick()
    root.mainloop()
