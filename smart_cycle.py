from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from datatypes import Result

from fixed_cycle import CAR_TYPES, PED_TYPES
from fsm import LAMPS, Config, FiniteStateMachine, Lamp, State

SETTLE_LIMIT = 3600.0


def simulate_smart(
        arrivals: list[dict],
        start_tod: time,
        cfg: Config = Config(),
        ped_min_remaining: float | None = None,
        dt: float = 0.1,
) -> list[Result]:
    if ped_min_remaining is not None:
        cfg = replace(cfg, ped_min_remaining=ped_min_remaining)

    # Group participants (input order is kept for the output)
    parsed = []
    for a in arrivals:
        kind, t = a["type"].lower(), float(a["arrival_time"])
        if t < 0:
            raise ValueError("arrival_time must be >= 0 (t = 0 is the simulation start)")
        if kind in PED_TYPES:
            parsed.append(("pedestrian", t))
        elif kind in CAR_TYPES:
            parsed.append(("vehicle", t))
        else:
            raise ValueError(f"unknown type '{a['type']}'")

    ped_times = []
    car_times = []
    last = 0.0
    for g, t in parsed:
        if g == "pedestrian":
            ped_times.append(t)
        else:
            car_times.append(t)
        if t > last:
            last = t
    ped_times.sort()
    car_times.sort()

    # run the fsm
    semaphore = FiniteStateMachine(cfg, t0=0.0, initial=State.CAR_GREEN)
    base = datetime.combine(date(2026, 10, 1), start_tod)
    i = j = k = 0
    while True:
        t = round(k * dt, 6)

        ped_count = 0
        while i < len(ped_times) and ped_times[i] <= t:  # all arrivals in this step
            i += 1
            ped_count += 1

        car_count = 0
        while j < len(car_times) and car_times[j] <= t:  # all arrivals in this step
            j += 1
            car_count += 1

        semaphore.step(t, (base + timedelta(seconds=t)).time(), ped_count, car_count)

        # Done once everyone has arrived, default leve tate: CAR_GREEN
        if (t >= last and i == len(ped_times) and j == len(car_times)
                and semaphore.state is State.CAR_GREEN and not semaphore.ped_pending):
            break
        if t > last + SETTLE_LIMIT:
            raise RuntimeError("FSM did not settle; check the configuration")
        k += 1

    # list of start times and corresponding states
    starts = [0.0]
    states = [State.CAR_GREEN]
    for tr in semaphore.history:
        starts.append(tr.t)
        states.append(tr.dst)
    ends = starts[1:] + [float("inf")]

    lamp_index = {"vehicle": 0, "pedestrian": 1}

    # Green windows per group
    def find_green_windows(lamp_idx: int, min_remaining: float):
        windows = []
        ends_adj = []
        for s, e, st in zip(starts, ends, states):
            if LAMPS[st][lamp_idx] is Lamp.GREEN and e - s > min_remaining:
                windows.append((s, e))
                ends_adj.append(e - min_remaining)
        return windows, ends_adj

    green_windows = {
        "pedestrian": find_green_windows(1, cfg.ped_min_remaining),
        "vehicle": find_green_windows(0, 0.0),
    }

    results = []
    for group, t in parsed:
        windows, ends_adj = green_windows[group]
        window_idx = bisect_right(ends_adj, t)  # first window still usable at t
        if window_idx == len(windows):
            raise RuntimeError("timeline ended before this arrival could be served")
        go = max(t, windows[window_idx][0])
        state = states[bisect_right(starts, t) - 1]
        results.append(Result(group, t, LAMPS[state][lamp_index[group]].value, go, go - t))
    return results
