from enum import Enum

class PlayerState(str, Enum):
    DAMAGE = "DAMAGE"
    NO_DAMAGE = "NO_DAMAGE"
    