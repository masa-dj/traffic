from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from datetime import time
from typing import Optional

CAR_MIN_GREEN = 30.0
PED_MIN_GREEN = 8.0
PED_MAX_GREEN = 12.0

# Fixed, mandatory times
CAR_YELLOW_TIME = 4.0
CLEAR_TO_PED_TIME = 1.0
CLEAR_TO_CAR_TIME = 10.0
CAR_RED_YELLOW_TIME = 2.0


# States for FSM, auto indexed
class State(Enum):
    CAR_GREEN = auto()
    CAR_YELLOW = auto()
    CLEAR_TO_PED = auto()
    PED_GREEN = auto()
    CLEAR_TO_CAR = auto()
    CAR_RED_YELLOW = auto()
    FLASHING_YELLOW = auto()


# Possible lights on the lamp
class Lamp(Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    RED_YELLOW = "red+yellow"
    YELLOW_FLASH = "flashing yellow"


# Mapping state to lights on the semaphores
# state: (car_lamp, pedestrian_lamp)
LAMPS = {
    State.CAR_GREEN: (Lamp.GREEN, Lamp.RED),
    State.CAR_YELLOW: (Lamp.YELLOW, Lamp.RED),
    State.CLEAR_TO_PED: (Lamp.RED, Lamp.RED),
    State.PED_GREEN: (Lamp.RED, Lamp.GREEN),
    State.CLEAR_TO_CAR: (Lamp.RED, Lamp.RED),
    State.CAR_RED_YELLOW: (Lamp.RED_YELLOW, Lamp.RED),
    State.FLASHING_YELLOW: (Lamp.YELLOW_FLASH, Lamp.YELLOW_FLASH),
}


@dataclass
class Transition:
    t: float
    src: State
    dst: State
    reason: str


# Create instance of "parameters package" to feed to machine
@dataclass(frozen=True)
class Config:
    night_start: time = time(22, 0)
    night_end: time = time(5, 0)

    w_count: float = 1.0
    w_wait: float = 0.1
    w_hold: float = 1.0
    ped_priority: float = 2.0
    car_window: float = 30.0
    ped_window: float = 5.0
    ped_max_wait: float = 90.0

    ped_min_remaining: float = 0.0

    # Default values to prevent errors
    def __post_init__(self):
        if min(self.w_count, self.w_wait, self.w_hold, self.ped_priority) < 0:
            raise ValueError("Weights cannot be negative")

    def is_night(self, tod: time) -> bool:
        if self.night_start > self.night_end:
            return tod >= self.night_start or tod < self.night_end
        return self.night_start <= tod < self.night_end


class FiniteStateMachine:
    def __init__(self, config: Config(), initial: State = State.CAR_RED_YELLOW, t0: float = 0):
        self.config = config
        self.state = initial
        self.entered_at = t0

        self.history: list[Transition] = []

        self.ped_waiting: deque[float] = deque()
        self.car_waiting: deque[float] = deque()
        self._ped_recent: deque[float] = deque()
        self._car_recent: deque[float] = deque()

    @property
    def lamps(self) -> tuple[Lamp, Lamp]:
        return LAMPS[self.state]

    @property
    def ped_pending(self) -> bool:
        return bool(self.ped_waiting)

    # Calculate pressures
    def _ped_pressure(self, t: float) -> float:
        if not self.ped_waiting:
            return 0.0
        c = self.config
        return c.ped_priority * (c.w_count * len(self.ped_waiting) + c.w_wait * (t - self.ped_waiting[0]))

    def _car_pressure(self, t: float) -> float:
        if not self.car_waiting:
            return 0.0
        c = self.config
        return c.w_count * len(self.car_waiting) + c.w_wait * (t - self.car_waiting[0])

    # Calculate if it is worth to keep green
    def _ped_hold(self) -> float:
        return self.config.ped_priority * self.config.w_hold * len(self._ped_recent)

    def _car_hold(self) -> float:
        return self.config.w_hold * len(self._car_recent)

    # Determine when to exit green car state
    def _exit_state_car_green(self, t: float, night: bool) -> Optional[str]:
        if night:
            return "night: pedestrian waiting"
        if t - self.ped_waiting[0] >= self.config.ped_max_wait:
            return "longest pedestrian wait reached cap"
        if self._ped_pressure(t) > self._car_hold():
            return "pedestrian pressure > car flow"
        return None

    # Determine when to exit green pedestrian state
    def _exit_state_ped_green(self, t: float, elapsed: float, night: bool) -> Optional[str]:
        if elapsed >= PED_MAX_GREEN:
            return "maximum pedestrian green reached"
        if night:
            return "night: shortest pedestrian green"
        hold = self._ped_hold()
        if hold == 0:
            return "no new pedestrians arriving"
        if self._car_pressure(t) > hold:
            return "car pressure > pedestrian flow"
        return None

    # Transition function
    def _go(self, t: float, dst: State, reason: str) -> None:
        if self.state is State.PED_GREEN:
            keep = self.config.ped_min_remaining
            self.ped_waiting = deque(a for a in self.ped_waiting if t - a <= keep)

        self.history.append(Transition(t, self.state, dst, reason))
        self.state = dst
        self.entered_at = t

        # Clear waiting deques
        if dst is State.PED_GREEN:
            self.ped_waiting.clear()
        elif dst is State.CAR_GREEN:
            self.car_waiting.clear()

    # Decision function
    def step(self, t: float, tod: time, ped_arrivals: int = 0, car_arrivals: int = 0) -> State:
        config = self.config
        s = self.state
        elapsed = t - self.entered_at
        night = config.is_night(tod)

        for i in range(int(ped_arrivals)):
            self.ped_waiting.append(t)
            self._ped_recent.append(t)
        for i in range(int(car_arrivals)):
            self._car_recent.append(t)
            if s is not State.CAR_GREEN:
                self.car_waiting.append(t)

        # Deque cleanup
        while self._ped_recent and t - self._ped_recent[0] > config.ped_window:
            self._ped_recent.popleft()
        while self._car_recent and t - self._car_recent[0] > config.car_window:
            self._car_recent.popleft()

        # 1. CAR_GREEN (changeable)
        if s is State.CAR_GREEN:
            if elapsed >= CAR_MIN_GREEN and self.ped_waiting:
                reason = self._exit_state_car_green(t, night)
                if reason:
                    self._go(t, State.CAR_YELLOW, reason)

        # 2. CAR_YELLOW (fixed 4 s)
        elif s is State.CAR_YELLOW:
            if elapsed >= CAR_YELLOW_TIME:
                self._go(t, State.CLEAR_TO_PED, "4s completed")

        # 3. CLEAR_TO_PED (fixed 1 s)
        elif s is State.CLEAR_TO_PED:
            if elapsed >= CLEAR_TO_PED_TIME:
                self._go(t, State.PED_GREEN, "1s completed")

        # 4. PED_GREEN (changeable)
        elif s is State.PED_GREEN:
            if elapsed >= PED_MIN_GREEN:
                reason = self._exit_state_ped_green(t, elapsed, night)
                if reason:
                    self._go(t, State.CLEAR_TO_CAR, reason)

        # 5. CLEAR_TO_CAR (fixed 10 s)
        elif s is State.CLEAR_TO_CAR:
            if elapsed >= CLEAR_TO_CAR_TIME:
                self._go(t, State.CAR_RED_YELLOW, "10s completed")

        # 6. CAR_RED_YELLOW (fixed 2 s)
        elif s is State.CAR_RED_YELLOW:
            if elapsed >= CAR_RED_YELLOW_TIME:
                self._go(t, State.CAR_GREEN, "2s completed")

        return self.state
