from pydantic import BaseModel
from app.common.CombinationResultState import CombinationResultState
from app.common.CombinationResultStateDetail import CombinationResultStateDetail
from app.common.GameScenario import GameScenario
from app.common.PlayerState import PlayerState
from app.common.ElementSymbols import ElementSymbols
from typing import Optional


class CombinationRes(BaseModel):
    material_a: ElementSymbols
    material_b: ElementSymbols
    result_1: Optional[ElementSymbols] = None
    result_2: Optional[ElementSymbols] = None
    result_state: CombinationResultState
    result_state_detail: CombinationResultStateDetail
    gold_melt: bool
    player_state: PlayerState
    scenario: GameScenario
    scenario_answer: ElementSymbols
    comment:str
