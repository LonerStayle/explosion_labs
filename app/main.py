from fastapi import FastAPI
from app.dto.CombinationReq import CombinationReq
from app.dto.HelpChatReq import HelpChatReq
from app.service.ai_service import AiService
from app.common.GameScenario import GameScenario
from app.dto.BaseRes import BaseRes
from app.dto.CombinationReq import CombinationReq
from app.dto.CombinationRes import CombinationRes
from app.common.ElementSymbols import ElementSymbols
from app.common.CombinationResultStateDetail import CombinationResultStateDetail
from app.common.CombinationResultState import CombinationResultState
from app.common.PlayerState import PlayerState
from app.common.GameScenario import GameScenario

app = FastAPI(
    title="미니프로젝트",
    description="API 서비스",
    version="1.0.0"
)

service = AiService()

@app.get("/health", tags=["헬스 체크"])
def health_check():
    return {"gggg":"ogggk"}



@app.post("/combination", response_model=BaseRes[CombinationRes], tags=["게임 API"])
async def combination_api(req: CombinationReq):
    """
    combination_api (비동기 함수)

    [목적]
    - 클라이언트(게임)로부터 두 개의 화학물질과 시나리오 정보를 받아 조합 결과를 반환합니다.
    - 실제 로직은 AiService의 combination_message 함수에 위임합니다.

    [파라미터]
    - req (CombinationReq): API 요청의 본문(body)에 담겨온 JSON 데이터입니다.
      FastAPI가 CombinationReq 모델에 따라 데이터의 유효성을 자동으로 검사하고 객체로 만들어줍니다.
    
    [반환]
    - BaseRes[CombinationRes]: AiService로부터 받은 CombinationRes 결과를 표준 응답 형식(BaseRes)으로
      감싸서 클라이언트에 JSON 형태로 반환합니다.
    """
    # 테스트용 가짜 데이터 대신, 실제 AI 서비스를 호출gka
    # 'await'는 비동기 함수인 service.combination_message가 끝날 때까지 기다리라는 의미입니다.
    result = await service.combination_message(req)
    
    # 성공적으로 처리된 결과를 표준 응답 래퍼(BaseRes)로 감싸서 반환합니다.
    return BaseRes(data=result)


@app.post("/help_chat")
async def help_chat_api(req: HelpChatReq):
    result = await service.help_message(req)
    return {"ai_message": result}
