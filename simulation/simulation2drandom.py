import random
import tkinter as tk

from intersection2 import Intersection
from controllers import FixedCycleController, SmartCycleController

TICK_MS = 1000

HORIZON = 3600
SEED = 1

MEAN_GAP = {
    ("pedestrian", "up"): 25,
    ("pedestrian", "down"): 25,
    ("vehicle", "right"): 10,
    ("vehicle", "left"): 12,
}

rng = random.Random(SEED)


def poisson(kind, direction, mean_gap, horizon):
    arrivals = []
    t = 0.0
    while t < horizon:
        t += rng.expovariate(1 / mean_gap)
        if t < horizon:
            arrivals.append({"type": kind, "direction": direction, "arrival_time": t})
    return arrivals


def generate_arrivals():
    arrivals = []
    for (kind, direction), gap in MEAN_GAP.items():
        arrivals += poisson(kind, direction, gap, HORIZON)
    arrivals.sort(key=lambda a: a["arrival_time"])
    return arrivals


def main():
    root = tk.Tk()
    root.title("Intersection Simulation")

    smart = Intersection(root, SmartCycleController(), title="Smart cycle")
    fixed = Intersection(root, FixedCycleController(), title="Fixed cycle")
    smart.grid(row=0, column=0, padx=10, pady=10)
    fixed.grid(row=0, column=1, padx=10, pady=10)

    time_label = tk.Label(root, text="t = 0 s", font=("Arial", 12))
    time_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

    arrivals = generate_arrivals()
    state = {"t": 0, "i": 0}

    def tick():
        t = state["t"]
        smart.update(t)
        fixed.update(t)

        while state["i"] < len(arrivals) and arrivals[state["i"]]["arrival_time"] < t + 1:
            arrival = arrivals[state["i"]]
            smart.add_arrival(dict(arrival))
            fixed.add_arrival(dict(arrival))
            state["i"] += 1

        time_label.config(text=f"t = {t} s")
        state["t"] += 1
        if state["t"] <= HORIZON:
            root.after(TICK_MS, tick)
        else:
            time_label.config(text=f"Finished (t = {HORIZON} s)")

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()