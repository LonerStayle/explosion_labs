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

@app.post("/combination", response_model=BaseRes[CombinationRes])
async def combination_api(req: CombinationReq):
    # fake_res = CombinationRes(
    #     material_a="Zn",
    #     material_b="HCl",
    #     result_1=ElementSymbols.H2,
    #     result_2="H2",
    #     result_state=CombinationResultState.SUCCESS,  # 예시 enum 값
    #     result_state_detail=CombinationResultStateDetail.NOTHING,  # 예시 enum 값
    #     gold_melt=False,
    #     player_state=PlayerState.NO_DAMAGE,              # 예시 enum 값
    #     scenario=GameScenario.COMBINE_HNO3HCL,        # 예시 enum 값
    #     scenario_answer=ElementSymbols.H2,
    #     comment = "질문해보세요"
    # )
    result = await service.combination_message(req)
    return BaseRes(data=result)


@app.post("/help_chat")
async def help_chat_api(req: HelpChatReq):
    result = await service.help_message(req)
    return {"ai_message": result}
