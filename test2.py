import asyncio
from app.service.ai_service import AiService
from app.dto.CombinationReq import CombinationReq
from app.common.ElementSymbols import ElementSymbols
from app.common.GameScenario import GameScenario
from app.dto.HelpChatReq import HelpChatReq


async def main():
    service = AiService()
    result01 = await service.combination_message(
        CombinationReq(
            material_a=ElementSymbols.HNO3,
            material_b=ElementSymbols.H2O,
            scenario=GameScenario.COMBINE_HNO3HCL,
        )
    )
    print(result01.json())  # JSON 변환

    # result02 = await service.help_message(
    #     HelpChatReq(
    #         select_material=ElementSymbols.HNO3,
    #         question="그다음 뭐해야돼??",
    #         scenario=GameScenario.COMBINE_HNO3HCL,
    #     )
    # )
    # print(result02)


asyncio.run(main())
