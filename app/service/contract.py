# 시연용 시나리오 매핑 (힌트 분기용)
SCENARIO_SOLUTIONS = {
    # 왕수 만들기 (변경 없음)
    "COMBINE_HNO3HCL": {
        "first": "HNO3",
        "second": "HCl",
        "order_required": True,
        "first_alias":  ["질", "강산", "자극 냄새"],
        "second_alias": ["염", "강산", "자극 냄새"],
    },

    # 왕수를 자물쇠1에 사용 (오브젝트 힌트)
    "USE_HNO3HCL": {
        "first": None, "second": None, "order_required": False,
        "object_alias": ["자물쇠1", "홈", "표식", "금속", "녹", "문양", "패턴"],
    },

    # K + H2O → KOH  (순서 중요: 금속 → 물)
    "COMBINE_KOH": {
        "first": "K",            # '반응성 금속'으로 암시
        "second": "H2O",         # '투명한 액체'로 암시
        "order_required": True,
        "first_alias":  ["반응성 금속", "은빛 금속", "부드러운 금속"],
        "second_alias": ["투명한 액체", "물", "액체"],
    },

    # KCN + HCl → HCN
    "COMBINE_HCN": {
        "first": "KCN",          # '특수 소금'으로 암시
        "second": "HCl",         # '강한 산'으로 암시
        "order_required": False,
        "first_alias":  ["특수 소금", "소금 같은 고체"],
        "second_alias": ["강한 산", "자극 냄새"],
    },

    # HCN + KOH → KCN
    "COMBINE_KCN": {
        "first": "HCN",          # '날카로운 냄새의 기체'로 암시
        "second": "KOH",         # '알칼리성 용액'으로 암시
        "order_required": False,
        "first_alias":  ["날카로운 냄새", "기체"],
        "second_alias": ["알칼리성 용액", "염기"],
    },

    # 청산가리를를 자물쇠2에 사용 (오브젝트 힌트)
    "USE_KCN": {
        "first": None, "second": None, "order_required": False,
        "object_alias": ["자물쇠2", "홈", "표식", "문양", "패턴"],
    },
}

import re
from typing import Optional
def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def compute_hint_case_advanced(scenario: str, material: Optional[str], question: Optional[str]) -> str:
    scenario_n = _norm(scenario)
    material_n = _norm(material)
    question_n = _norm(question)

    sol = SCENARIO_SOLUTIONS.get(scenario)
    if not sol:
        return "FREE_HINT"

    # USE_* 시나리오: 단계 힌트가 아니라 자물쇠/오브젝트 힌트만
    if scenario_n.startswith("use_"):
        return "FREE_HINT"

    # 위험 제조 시나리오: 항상 완곡 회피(안전 측면에서 직접 유도 금지)
    if sol.get("danger"):
        return "FREE_HINT"

    first  = (sol.get("first")  or "").lower() if sol.get("first")  else None
    second = (sol.get("second") or "").lower() if sol.get("second") else None
    order_required = sol.get("order_required", False)

    first_alias  = [a.lower() for a in sol.get("first_alias", [])]
    second_alias = [a.lower() for a in sol.get("second_alias", [])]

    asked_first  = any(a in question_n for a in first_alias)  if first_alias  else False
    asked_second = any(a in question_n for a in second_alias) if second_alias else False

    # A) 첫 재료 미선택
    if not material_n:
        if asked_second and first and second:
            return "SECOND_WHILE_FIRST_WRONG"
        if first:
            return "FIRST_WRONG_GUIDE"
        return "FREE_HINT"

    # B) 첫 번째 정답이 맞음  ← ★ 빠져 있던 분기 복구
    if first and material_n == first:
        if second:
            return "SECOND_HINT"   # 두 번째를 암시
        return "CLOSE"             # 단일 타깃형(두 번째 없음)

    # C) 두 번째를 먼저 선택
    if second and material_n == second:
        if order_required:
            return "FIRST_ORDER_NUDGE"  # 순서 교정
        return "CLOSE"

    # D) 첫 재료부터 오답
    if (not first or material_n != first) and (not second or material_n != second):
        if asked_second and first and second:
            return "SECOND_WHILE_FIRST_WRONG"
        if first:
            return "FIRST_WRONG_GUIDE"
        return "FREE_HINT"

    # E) 기타
    return "FREE_HINT"

def is_chemical_question(question: str) -> bool:
    science_keywords = [
        "화합물", "원소", "원자", "이온", "전자", "기체", "용액",
        "산", "염기", "중화", "반응", "조합", "혼합",
        "실험", "실험실", "시험관", "비커", "물질", "분자",
        "색깔", "가스", "버블", "거품", "침전", "열", "냄새","화학", "도와","도움"
        "정답", "힌트", "플레이어", "선택", "시나리오", "어떻","어떡", "뭐"
    ]
    return any(kw in question for kw in science_keywords)
