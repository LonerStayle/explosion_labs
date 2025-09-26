from enum import Enum

class CombinationResultStateDetail(str, Enum):
    POSION_GAS = "POSION_GAS"
    EXPLOSION = "EXPLOSION"
    NOTHING = "NOTHING"