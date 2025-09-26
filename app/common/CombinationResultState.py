from enum import Enum

class CombinationResultState(str, Enum):
    SUCCESS = "SUCCESS"
    BAD = "BAD"
    NOTHING = "NOTHING"