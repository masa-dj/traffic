from __future__ import annotations
from datatypes import Result

# One full cycle
CYCLE_PLAN = [
    (0, 68, "green", "red"),
    (68, 72, "yellow", "red"),
    (72, 73, "red", "red"),
    (73, 81, "red", "green"),
    (81, 91, "red", "red"),
    (91, 93, "red+yellow", "red"),
]
CYCLE = CYCLE_PLAN[-1][1]

# Which lamps let someone go
CAR_GO = {"green"}
PED_GO = {"green"}

# Grouping based on type of participant into 2 groups
PED_TYPES = {"person", "pedestrian"}
CAR_TYPES = {"car", "vehicle", "truck", "bus", "motorcycle", "bicycle"}


# Lookup the cycle to determine green light windows for respected groups
def go_windows(lamp_index, go_lamps):
    windows = []
    for start, end, car, ped in CYCLE_PLAN:
        lamp = car if lamp_index == 0 else ped
        if lamp in go_lamps:
            windows.append((start, end))
    return windows


CAR_WINDOWS = go_windows(0, CAR_GO)
PED_WINDOWS = go_windows(1, PED_GO)


# To determine if participant can go, we need to find the state of semaphore upon his arrival
def lamps_at(t: float, offset: float = 0.0) -> tuple[str, str]:
    p = (t - offset) % CYCLE
    for start, end, car, ped in CYCLE_PLAN:
        if start <= p < end:
            return car, ped
    raise AssertionError("cycle plan has a gap")


# Determine if participant can go based on cycle status
def go_time(t: float, windows, offset: float = 0.0, min_remaining: float = 0.0) -> float:
    p = (t - offset) % CYCLE
    base = t - p  # start of current cycle
    for k in (0, 1):
        for s, e in windows:
            start = base + k * CYCLE + s
            end = base + k * CYCLE + e - min_remaining
            candidate = max(t, start)
            if candidate < end:
                return candidate
    raise ValueError("no usable green window; check min_remaining and the cycle plan")


# Take list of participants, group accordingly and get waiting times
def simulate(arrivals: list[dict], offset: float = 0.0, ped_min_remaining: float = 0.0) -> list[Result]:
    results = []
    for a in arrivals:
        kind, t = a["type"].lower(), float(a["arrival_time"])
        car_lamp, ped_lamp = lamps_at(t, offset)
        if kind in PED_TYPES:
            group, lamp = "pedestrian", ped_lamp
            g = go_time(t, PED_WINDOWS, offset, ped_min_remaining)
        elif kind in CAR_TYPES:
            group, lamp = "vehicle", car_lamp
            g = go_time(t, CAR_WINDOWS, offset)
        else:
            raise ValueError(f"unknown type '{a['type']}'")
        results.append(Result(group, t, lamp, g, g - t))
    return results
