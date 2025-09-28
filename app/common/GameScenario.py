from enum import Enum


class GameScenario(str, Enum):
    COMBINE_HNO33HCL = "COMBINE_HNO33HCL"
    USE_HNO3HCL = "USE_HNO3HCL"
    COMBINE_KOH = "COMBINE_KOH"
    COMBINE_KCL = "COMBINE_KCL"
    COMBINE_KCN = "COMBINE_KCN"
    USE_KCN = "USE_KCN"
