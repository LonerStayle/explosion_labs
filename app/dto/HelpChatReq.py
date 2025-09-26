from app.common.ElementSymbols import ElementSymbols
from pydantic import BaseModel
from app.common.GameScenario import GameScenario

class HelpChatReq(BaseModel): 
    select_material: ElementSymbols
    question:str
    scenario: GameScenario
    
