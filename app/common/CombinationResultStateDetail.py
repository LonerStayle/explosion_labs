from enum import Enum

class CombinationResultStateDetail(str, Enum):
    POISON_GAS = "POISON_GAS"
    EXPLOSION = "EXPLOSION"
    NOTHING = "NOTHING"