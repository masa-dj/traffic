from dataclasses import dataclass
from enum import Enum, auto

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
