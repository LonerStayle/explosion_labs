from enum import Enum

class ElementSymbols(str, Enum):
    HNO3 = "HNO3" # 진한 질산(HNO₃)
    HCL_CONC = "HCL_CONC" # 진한 염산(HCL)
    KCN = "KCN" # 시안화 칼륨(KCN)
    HCL = "HCL" # 염산(HCL)
    H20 = "H20" # 물(H₂O)
    K = "K" # 칼륨(K)
    HCN = "HCN" # 시안화 수소(HCN)
    KOH = "KOH" # 수산화 칼륨(KOH)
    I2 = "I2" # 아이오딘(I₂)
    NH3H20 = "NH3H20" # 암모니아수(NH₃·H₂O)
    CA = "CA" # 칼슘(Ca)
    MG = "MG" # 마그네슘(Mg)
    FE = "FE" # 철(Fe)
    ZN = "ZN" # 아연(Zn)
    HNO33HCL = "HNO33HCL" # 왕수(HNO₃ + 3HCL)
    H2 = "H2" # 수소(H₂)기체
    KCL = "KCL" # 염화 칼륨(KCL)
    H2O = "H2O" # 물(H₂O)
    NI3NH3 = "NI3NH3" # 질소 삼아이오딘-암모니아 착물(NI₃·NH₃)
    HI = "HI" # 아이오딘화 수소(HI)
