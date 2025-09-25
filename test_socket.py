import asyncio
import websockets
import json

async def test_ws():
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

asyncio.run(test_ws())