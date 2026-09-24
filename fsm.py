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
    def __init__(self, config: Config(), initial: State = State.CAR_RED_YELLOW, t0: float=0):
        self.config = config
        self.state = initial
        self.entered_at = t0

        self.history: list[Transition] = []

        self.ped_waiting: deque[float] = deque()
        self.car_waiting: deque[float] = deque()

    @property
    def lamps(self) -> tuple[Lamp, Lamp]:
        return LAMPS[self.state]

    # Determine when to exit green car state
    def _exit_state_car_green(self, t: float, night: bool) -> Optional[str]:
        return "left"

    # Determine when to exit green pedestrian state
    def _exit_state_ped_green(self, t: float, elapsed: float, night: bool) -> Optional[str]:
        if elapsed >= PED_MAX_GREEN:
            return "maximum pedestrian green reached"
        return "left"

    # Transition function
    def _go(self, t: float, dst: State, reason: str) -> None:
        self.history.append(Transition(t, self.state, dst, reason))
        self.state = dst
        self.entered_at = t

    # Decision function
    def step(self, t: float, tod: time, ped_arrivals: int = 0, car_arrivals: int = 0) -> State:
        config = self.config
        s = self.state
        elapsed = t - self.entered_at
        night = config.is_night(tod)

        for i in range(int(ped_arrivals)):
            self.ped_waiting.append(t)
        for i in range(int(car_arrivals)):
            if s is not State.CAR_GREEN:
                self.car_waiting.append(t)

        # 1. CAR_GREEN (changeable)
        if s is State.CAR_GREEN:
            self._go(t, State.CAR_YELLOW, "reason")

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
            self._go(t, State.CLEAR_TO_CAR, "reason")

        # 5. CLEAR_TO_CAR (fixed 10 s)
        elif s is State.CLEAR_TO_CAR:
            if elapsed >= CLEAR_TO_CAR_TIME:
                self._go(t, State.CAR_RED_YELLOW, "10s completed")

        # 6. CAR_RED_YELLOW (fixed 2 s)
        elif s is State.CAR_RED_YELLOW:
            if elapsed >= CAR_RED_YELLOW_TIME:
                self._go(t, State.CAR_GREEN, "2s completed")

        return self.state
