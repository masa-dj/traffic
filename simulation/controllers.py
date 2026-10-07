from datetime import time

from fsm import FiniteStateMachine, Config, LAMPS


CYCLE_PLAN = [
    (0, 68, "green", "red"),
    (68, 72, "yellow", "red"),
    (72, 73, "red", "red"),
    (73, 81, "red", "green"),
    (81, 91, "red", "red"),
    (91, 93, "red+yellow", "red"),
]

# Which lamps let someone go
CAR_GO = {"green"}
PED_GO = {"green"}


class FixedCycleController:
    def __init__(self, plan=None):
        self.plan = plan if plan is not None else CYCLE_PLAN
        self.cycle = self.plan[-1][1]

    def state_at(self, t):
        t_mod = t % self.cycle
        for start, end, car, ped in self.plan:
            if start <= t_mod < end:
                return car, ped
        # Fallback
        start, end, car, ped = self.plan[-1]
        return car, ped


class SmartCycleController:
    def __init__(self, config: Config = None, tod: time = time(12, 0)):
        self.fsm = FiniteStateMachine(config or Config())
        self.tod = tod
        self._pending_ped_arrivals = 0
        self._pending_car_arrivals = 0

    def notify_arrival(self, kind):
        if kind == "pedestrian":
            self._pending_ped_arrivals += 1
        elif kind == "vehicle":
            self._pending_car_arrivals += 1
        else:
            raise ValueError(f"Unknown arrival kind: {kind!r}")

    def state_at(self, t):
        state = self.fsm.step(
            t,
            self.tod,
            ped_arrivals=self._pending_ped_arrivals,
            car_arrivals=self._pending_car_arrivals,
        )
        self._pending_ped_arrivals = 0
        self._pending_car_arrivals = 0

        car_lamp, ped_lamp = LAMPS[state]
        return car_lamp.value, ped_lamp.value
