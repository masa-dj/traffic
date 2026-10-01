from datatypes import Result
import numpy as np


# Calculate statistical parameters
def summarize(results: list[Result]) -> dict[str, dict]:
    out = {}
    for group in ("pedestrian", "vehicle"):
        waits = []
        for r in results:
            if r.group == group:
                waits.append(r.wait)
        if len(waits) == 0:
            continue
        w = np.array(waits)
        out[group] = {
            "n": int(w.size),
            "mean": float(w.mean()),
            "median": float(np.median(w)),
            "p95": float(np.percentile(w, 95)),
            "max": float(w.max()),
            "share_waited": float((w > 0).mean()),
        }
    return out
