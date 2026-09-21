from dataclasses import dataclass


@dataclass
class Result:
    group: str
    arrival_time: float
    lamp_at_arrival: str
    go_time: float
    wait: float


@dataclass
class VehicleRecord:
    id: int
    type: str
    appeared_at: float
    left: float
    waiting_time: float


@dataclass
class PedestrianRecord:
    id: int
    type: str
    appeared_at: float
    left: float
    waiting_time: float
