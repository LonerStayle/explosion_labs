from app.common.ElementSymbols import ElementSymbols
from pydantic import BaseModel

class HelpChatReq(BaseModel): 
    select_symbols:ElementSymbols
    question:str
