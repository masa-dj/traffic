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
        _, _, car, ped = self.plan[-1]
        return car, ped


class SmartCycleController:
    def __init__(self):
        self._fallback = FixedCycleController()

    def state_at(self, t):
        # TODO: add the right logic
        return self._fallback.state_at(t)
