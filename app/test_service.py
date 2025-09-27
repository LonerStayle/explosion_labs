import asyncio

# --- [수정] 모든 import 경로 앞에 'app.'을 붙여줍니다. ---
from app.service.ai_service import AiService
from app.dto.CombinationReq import CombinationReq
from app.dto.HelpChatReq import HelpChatReq
from app.common.ElementSymbols import ElementSymbols
from app.common.GameScenario import GameScenario


async def main():
    """
    AiService의 함수들을 테스트하기 위한 메인 비동기 함수.
    """
    print("AiService 초기화 시작...")
    service = AiService()
    print("AiService 초기화 완료.\n")

    # --- 1. combination_message 함수 테스트 ---
    print("--- 1. combination_message 테스트 시작 ---")
    try:
        combination_request = CombinationReq(
            material_a=ElementSymbols.HNO3,
            material_b=ElementSymbols.HCL_CONC,
            scenario=GameScenario.COMBINE_HNO3HCL,
        )
        print(f"요청: {combination_request.model_dump()}")

        result_combination = await service.combination_message(combination_request)

        print("\n[결과]")
        print(result_combination.model_dump_json(indent=2))

    except Exception as e:
        print(f"combination_message 테스트 중 에러 발생: {e}")
    print("--- 1. combination_message 테스트 종료 ---\n")


    # --- 2. help_message 함수 테스트 ---
    print("--- 2. help_message 테스트 시작 ---")
    try:
        help_request = HelpChatReq(
            select_material=ElementSymbols.HNO3.value,
            question="왕수는 어떻게 만드나요?",
            scenario=GameScenario.COMBINE_HNO3HCL.value,
        )
        print(f"요청: {help_request.model_dump()}")

        result_help = await service.help_message(help_request)

        print("\n[결과]")
        print(f"AI 응답: {result_help}")

    except Exception as e:
        print(f"help_message 테스트 중 에러 발생: {e}")
    print("--- 2. help_message 테스트 종료 ---")


if __name__ == "__main__":
    asyncio.run(main())