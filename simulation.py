from __future__ import annotations
from analysis import summarize
import random
from datetime import time
from fixed_cycle import simulate
from smart_cycle import simulate_smart

rng = random.Random(1)


# Poissons distibution
def poisson(kind: str, mean_gap: float, horizon: float) -> list[dict]:
    arrivals = []
    t = 0.0
    while t < horizon:
        # gaps between arrivals are exponentially distributed
        gap = rng.expovariate(1 / mean_gap)
        t = t + gap
        if t < horizon:
            arrivals.append({"type": kind, "arrival_time": t})

    return arrivals


# Run both controllers on the same arrivals and
def compare(title: str, arrivals: list[dict], start_tod: time) -> None:
    fixed = summarize(simulate(arrivals))
    smart = summarize(simulate_smart(arrivals, start_tod))

    print(f"\n{title}")
    for group in ("pedestrian", "vehicle"):
        if group not in fixed:
            continue
        for name, stats in (("fixed", fixed), ("smart", smart)):
            s = stats[group]
            print(f"  {group:<11} {name:<6} n={s['n']:<5} mean={s['mean']:6.2f}s  median={s['median']:6.2f}s  "
                  f"p95={s['p95']:6.2f}s  max={s['max']:6.2f}s  waited={s['share_waited']:.0%}")
        diff = smart[group]["mean"] - fixed[group]["mean"]
        pct = f" ({diff / fixed[group]['mean']:+.0%})" if fixed[group]["mean"] > 0 else ""
        print(f"  {'':<11} mean wait with smart cycle: {diff:+.2f}s{pct}")


if __name__ == "__main__":
    HOUR = 3600.0

    # Day: pedestrian every 20 s, car every 4 s
    arrivals = poisson("person", 20.0, HOUR) + poisson("car", 4.0, HOUR)
    compare("Midday from 12:00, busy, 1 h", arrivals, time(12, 0))

    # Night window: pedestrian every 5 min, car every 1 min
    arrivals = poisson("person", 300.0, 6 * HOUR) + poisson("car", 60.0, 6 * HOUR)
    compare("Night from 23:00, quiet, 6 h", arrivals, time(23, 0))
