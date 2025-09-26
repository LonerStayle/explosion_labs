from app.common.ElementSymbols import ElementSymbols
from app.common.GameScenario import GameScenario
from pydantic import BaseModel

class CombinationReq(BaseModel):
    material_a: ElementSymbols
    material_b: ElementSymbols
    scenario: GameScenario
    
    class Config:
        json_schema_extra = {
            "example": {
                "material_a": "K",
                "material_b": "H2O",
                "scenario" : "Combine_HNO3HCL"
            }
        }