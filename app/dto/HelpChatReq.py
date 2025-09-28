from app.common.ElementSymbols import ElementSymbols
from pydantic import BaseModel
from app.common.GameScenario import GameScenario
from typing import Optional

class HelpChatReq(BaseModel): 
    select_material: Optional[ElementSymbols] = None
    question:str
    scenario: GameScenario
    
