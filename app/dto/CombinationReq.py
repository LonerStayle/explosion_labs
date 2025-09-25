from app.common.ElementSymbols import ElementSymbols
from pydantic import BaseModel
from enum import Enum
from typing import List

class CombinationReq(BaseModel):
    symbols: List[ElementSymbols]
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbols": [
                    "Hydrogen",
                    "Helium",
                    "Lithium"
                ]
            }
        }