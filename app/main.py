from fastapi import FastAPI, WebSocket
from app.dto.CombinationReq import CombinationReq
from app.dto.BaseRes import BaseRes
import json
from app.service.ai_service import AiService


app = FastAPI(
    title="미니프로젝트",
    description="API 서비스",
    version="1.0.0"
)

@app.get("/health", tags=["헬스 체크"])
def health_check():
    return {"status":"ok"}


@app.websocket("/ws/combination")
async def combination(ws: WebSocket):
    await ws.accept()
    service = AiService()
    try:
        while True:
            raw = await ws.receive_text()   
            obj = json.loads(raw)           
            req = CombinationReq(**obj)      
            async for token in service.combination_message(req):
                await ws.send_text(token)

            await ws.close(code=1000, reason="done")
    except Exception as e: 
        print("WS Error", e)
    finally:
        await ws.close(code=1000, reason="server shutdown")  


@app.websocket("/ws/help_chat")
async def chat(ws:WebSocket):
    await ws.accept()
    service = AiService()    
    try:
        while True:
            text = await ws.receive_text()
            async for token in service.help_message(text):
                await ws.send_text(token)

    except Exception as e: 
        print("WS Error", e)
    finally:
        await ws.close(code=1000, reason="server shutdown")
    