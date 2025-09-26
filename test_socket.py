import asyncio
import websockets
import json

async def test_ws_combination():
    uri = "ws://127.0.0.1:8000/ws/combination"

    async with websockets.connect(uri) as ws:
        req = {"symbols": ["Hydrogen", "Helium", "Lithium"]}
        await ws.send(json.dumps(req))

        try:
            while True:
                msg = await ws.recv()
                print("📩 서버 응답:", msg)
            
        except websockets.exceptions.ConnectionClosedOK:
            print("🔌 연결 종료")

async def test_ws_help_chat():
    uri = "ws://127.0.0.1:8000/ws/help_chat"
    text = input("질문을 입력하세요: ")
    async with websockets.connect(uri) as ws:    
        await ws.send(text)
        try:
            while True:
                msg = await ws.recv()
                print("📩 서버 응답:", msg)
            
        except websockets.exceptions.ConnectionClosedOK:
            print("🔌 연결 종료")


asyncio.run(test_ws_help_chat())