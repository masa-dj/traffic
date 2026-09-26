import tkinter as tk

from intersection import Intersection
from controllers import FixedCycleController, SmartCycleController

TICK_MS = 1000


def main():
    root = tk.Tk()
    root.title("Intersection Simulation")

    smart = Intersection(root, SmartCycleController(), title="Smart cycle")
    fixed = Intersection(root, FixedCycleController(), title="Fixed cycle")

    smart.grid(row=0, column=0, padx=10, pady=10)
    fixed.grid(row=0, column=1, padx=10, pady=10)

    t = 0

    def tick():
        nonlocal t
        smart.update(t)
        fixed.update(t)
        t += 1
        root.after(TICK_MS, tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()