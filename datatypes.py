from dataclasses import dataclass


@dataclass
class Result:
    group: str
    arrival_time: float
    lamp_at_arrival: str
    go_time: float
    wait: float
