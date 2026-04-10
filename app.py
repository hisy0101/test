import streamlit as st
from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import plotly.graph_objects as go
from textwrap import dedent

# =========================
# 1. 모델명 / 정책 상수
# =========================
MODEL_NAME = "gemini-2.5-flash"

WEIGHTS = {
    "concept_understanding": 0.4,
    "logical_writing": 0.3,
    "term_usage": 0.3,
}

LEVEL_RULES = [
    (90, "매우 우수"),
    (75, "우수"),
    (50, "보통"),
    (0, "노력 요함"),
]

FINAL_GRADE_LABELS = {
    "매우 우수": "전교 1등 포스 ✨",
    "우수": "오~ 좀 치는데? 👍",
    "보통": "가능성이 보여 😊",
    "노력 요함": "선배랑 특훈 가자 💪",
}

LEVEL_ORDER = {
    "노력 요함": 1,
    "보통": 2,
    "우수": 3,
    "매우 우수": 4,
}

# 조건 충족에 따른 최종 등급 상한 정책
# - X 1개 이상 → 최대 보통
# - X는 없고 △ 1개 이상 → 최대 우수
# - 전부 ○ → 상한 없음
CONDITION_CAP_POLICY = {
    "X": "보통",
    "△": "우수",
    "○": "매우 우수",
}


# =========================
# 2. Pydantic 결과 구조
# =========================
class CategoryResult(BaseModel):
    score: int = Field(ge=0, le=100, description="항목별 점수 (0~100)")
    level: Literal["매우 우수", "우수", "보통", "노력 요함"] = Field(
        description="항목 수준"
    )
    feedback: str = Field(
        min_length=10,
        description="학생이 바로 이해할 수 있는 구체적 피드백. 다음에 어떻게 고치면 좋을지 행동 제안 포함."
    )


class IndividualCondition(BaseModel):
    name: str = Field(min_length=1, description="조건명")
    status: Literal["○", "△", "X"] = Field(description="조건 충족 여부")
    reason: str = Field(
        min_length=5,
        description="해당 판정 이유. 학생이 이해할 수 있는 쉬운 설명."
    )


class AnalysisReport(BaseModel):
    concept_understanding: CategoryResult = Field(description="개념 이해 평가")
    logical_writing: CategoryResult = Field(description="논리적 서술 평가")
    term_usage: CategoryResult = Field(description="용어 사용 평가")
    individual_conditions: List[IndividualCondition] = Field(
        description="각 조건별 판정 결과"
    )
    encouragement: str = Field(
        min_length=3,
        description="짧고 따뜻한 한 줄 격려"
    )
    overall_summary: str = Field(
        min_length=10,
        description="답안 전체에 대한 요약 총평"
    )


# =========================
# 3. 그래프 상태 정의
# =========================
class GradingState(TypedDict):
    question: str
    conditions: str
    reference: str
    keywords: str
    student_answer: str
    api_key: str
    analysis_result: Optional[AnalysisReport]


# =========================
# 4. 유틸 함수
# =========================
def get_level_from_score(score: int) -> str:
    for threshold, level in LEVEL_RULES:
        if score >= threshold:
            return level
    return "노력 요함"


def get_star_rating(level: str) -> str:
    mapping = {
        "매우 우수": "★★★★",
        "우수": "★★★☆",
        "보통": "★★☆☆",
        "노력 요함": "★☆☆☆",
    }
    return mapping.get(level, "☆☆☆☆")


def cap_level_by_conditions(level: str, conditions: List[IndividualCondition]) -> str:
    statuses = [c.status for c in conditions]

    if "X" in statuses:
        cap = CONDITION_CAP_POLICY["X"]
    elif "△" in statuses:
        cap = CONDITION_CAP_POLICY["△"]
    else:
        cap = CONDITION_CAP_POLICY["○"]

    return cap if LEVEL_ORDER[level] > LEVEL_ORDER[cap] else level


def calculate_final_score(report: AnalysisReport) -> int:
    weighted_score = (
        report.concept_understanding.score * WEIGHTS["concept_understanding"]
        + report.logical_writing.score * WEIGHTS["logical_writing"]
        + report.term_usage.score * WEIGHTS["term_usage"]
    )
    return round(weighted_score)


def validate_student_answer(answer: str) -> Optional[str]:
    if not answer.strip():
        return "답안을 입력해 주세요."
    if len(answer.strip()) < 5:
        return "답안이 너무 짧아요. 조금 더 자세히 작성해 주세요."
    return None


def get_condition_summary(conditions: List[IndividualCondition]):
    if not conditions:
        return {
            "level": "보통",
            "feedback": "확인할 조건 정보가 없어서 조건 충족 여부를 판단하기 어려워. 문항에 제시된 요구 조건을 먼저 정리해 두면 더 정확하게 볼 수 있어."
        }

    partial = sum(1 for c in conditions if c.status == "△")
    missing = sum(1 for c in conditions if c.status == "X")

    if missing == 0 and partial == 0:
        level = "매우 우수"
        feedback = "문항에서 요구한 조건을 빠짐없이 잘 반영했어. 키워드, 형식, 분량 같은 지시사항을 놓치지 않고 답안에 자연스럽게 녹여 낸 점이 좋아."
    elif missing == 0 and partial >= 1:
        level = "우수"
        feedback = f"주요 조건은 대부분 잘 반영했어. 다만 {partial}개 조건은 조금 애매하게 드러나서, 답안에 요구 조건이 보이도록 표현을 한 번만 더 분명하게 써 주면 더 완성도 높아질 거야."
    elif missing == 1:
        level = "보통"
        feedback = "핵심 조건은 어느 정도 반영했지만, 빠진 조건이 있어서 점수가 조금 아까워. 문항을 쓴 뒤에 '조건을 다 넣었는지' 마지막으로 체크하는 습관을 들이면 훨씬 좋아질 거야."
    else:
        level = "노력 요함"
        feedback = "문항에서 요구한 조건이 여러 개 빠져 있어서 답안 완성도가 낮아 보였어. 답을 쓰기 전에 조건을 먼저 표시해 두고, 쓴 뒤에 하나씩 확인하는 방식으로 정리해 보면 좋아."

    return {
        "level": level,
        "feedback": feedback
    }


# =========================
# 5. 분석 노드 (테스트용 Mock 버전)
# =========================
def analysis_node(state: GradingState):
    import time
    time.sleep(2)  # 분석 중인 느낌

    ans_len = len(state["student_answer"])

    mock_report = AnalysisReport(
        concept_understanding=CategoryResult(
            score=95 if ans_len > 30 else 60,
            level="매우 우수" if ans_len > 30 else "보통",
            feedback="말씀하신 대로 '합력'의 개념을 정확하게 짚었어. 큰 힘에서 작은 힘을 빼는 원리를 답안에 잘 녹여 낸 점이 훌륭해."
        ),
        logical_writing=CategoryResult(
            score=88 if ans_len > 20 else 40,
            level="우수" if ans_len > 20 else "노력 요함",
            feedback="계산 과정이 논리적으로 잘 서술되었어. 다만 결론에서 방향을 한 번 더 강조해 주면 완벽한 답안이 될 거야."
        ),
        term_usage=CategoryResult(
            score=90,
            level="매우 우수",
            feedback="'N(뉴턴)' 단위와 '합력'이라는 용어를 아주 적절하게 사용했어. 과학적 표현력이 상당히 지적인걸?"
        ),
        individual_conditions=[
            IndividualCondition(
                name="계산 과정 포함",
                status="○" if "350" in state["student_answer"] or "150" in state["student_answer"] else "X",
                reason="계산식(350-150) 또는 계산 과정이 드러나 있는지 기준으로 확인했어."
            ),
            IndividualCondition(
                name="크기와 방향 명시",
                status="△",
                reason="크기(200N)는 잘 썼지만, 방향 설명이 조금 더 또렷하면 좋아."
            ),
            IndividualCondition(
                name="용어 사용",
                status="○",
                reason="'합력' 또는 '알짜힘' 용어를 정확히 사용했어."
            )
        ],
        encouragement="답안의 방향은 아주 잘 잡았어! 조금만 더 구체적으로 쓰면 전교 1등도 문제없겠어.",
        overall_summary="전반적으로 개념에 대한 이해도가 높고 논리적인 답안이야. 조건 중 '방향' 부분만 보완하면 완벽해질 것 같아. 선배가 보기엔 가능성이 무궁무진해!"
    )

    return {"analysis_result": mock_report}


# =========================
# 6. 그래프 구축
# =========================
workflow = StateGraph(GradingState)
workflow.add_node("grading", analysis_node)
workflow.set_entry_point("grading")
workflow.add_edge("grading", END)
analysis_app = workflow.compile()


# =========================
# 7. Streamlit UI
# =========================
st.set_page_config(page_title="AI 사회, 과학 서술형 Master", layout="wide")

st.markdown("""
<style>
section.main > div {
    padding-top: 0.2rem !important;
}
section.main div.block-container {
    padding-top: 0.35rem !important;
    padding-bottom: 0.5rem !important;
}
html, body, [class*="css"] {
    font-size: 100% !important;
}
p, li, div {
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
}
div[data-testid="stHeading"] h1,
div[data-testid="stHeadingWithActionElements"] h1 {
    font-size: inherit !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}
[data-testid="stCaptionContainer"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stCaptionContainer"] p {
    font-size: 1.08rem !important;
    color: #666 !important;
    line-height: 1.35 !important;
    margin-top: 0.15rem !important;
    margin-bottom: 0 !important;
}
h2 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}
h3 {
    font-size: 1.28rem !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] * {
    font-size: 1.02rem !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    line-height: 1.45 !important;
}
textarea, input {
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}
button p {
    font-size: 1.06rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.02rem !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}
.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
}
textarea, input {
    border-radius: 12px !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.03);
}
.stButton > button {
    border-radius: 12px !important;
    border: none !important;
    background: linear-gradient(90deg, #2563EB 0%, #4F46E5 100%) !important;
    color: white !important;
    box-shadow: 0 8px 18px rgba(37,99,235,0.22);
}
[data-testid="stAlertContainer"] p {
    font-size: 1rem !important;
    line-height: 1.55 !important;
}

/* 수정 위치 A: 점수 카드 공통 스타일 */
.score-card-wrap {
    display:flex;
    gap:16px;
    margin-bottom:18px;
}
.score-card {
    flex:1;
    background:#FFFFFF;
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:16px 18px;
    box-shadow:0 4px 12px rgba(15,23,42,0.04);
    height:120px;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
}
.score-label {
    font-size:0.95rem;
    color:#374151;
    font-weight:700;
}
.score-value {
    font-size:1.35rem;
    color:#111827;
    font-weight:800;
    line-height:1.2;
}
.score-sub {
    display:inline-block;
    width:fit-content;
    padding:6px 10px;
    border-radius:999px;
    background:#ECFDF3;
    color:#047857;
    font-size:0.9rem;
    font-weight:700;
}
.total-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:20px 22px;
    box-shadow:0 6px 18px rgba(15, 23, 42, 0.05);
    margin-bottom:18px;
}
.total-card-badge {
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    background:#F3F4F6;
    color:#374151;
    font-size:0.82rem;
    font-weight:700;
    margin-bottom:12px;
}
.total-rank {
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    background:#ECFDF3;
    color:#047857;
    font-size:0.9rem;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.markdown(dedent("""
<div style="
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    background:#EEF2FF;
    color:#4338CA;
    font-size:0.86rem;
    font-weight:700;
    margin-bottom:10px;
">
    Preview · AI 평가 결과 시뮬레이션
</div>
"""), unsafe_allow_html=True)

st.markdown(dedent("""
<div style="
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    gap:12px;
    margin:0;
    padding:0;
    line-height:1;
">
    <span style="
        color:#289c46;
        white-space:nowrap;
        font-size:2.4rem;
        font-weight:900;
        line-height:1;
    ">Mbest</span>
    <span style="
        font-size:2.4rem;
        font-weight:900;
        line-height:1;
        letter-spacing:-0.01em;
    ">AI 사, 과 서술형 Master 👑</span>
</div>
"""), unsafe_allow_html=True)

st.caption("AI가 항목별 분석과 피드백을 생성하고, 최종 점수와 등급은 시스템 계산 로직으로 산출합니다.")
st.divider()

with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    st.info(f"가동 모델: {MODEL_NAME}")

    st.divider()
    st.subheader("📘 평가 기준")
    st.markdown(
        "- 개념 이해: 40%\n"
        "- 논리적 서술: 30%\n"
        "- 용어 사용: 30%\n"
        "- 조건 충족: 최종 등급 상한 적용"
    )

    st.divider()
    st.subheader("📏 등급 상한 정책")
    st.markdown(
        "- X 1개 이상 → 최대 **보통**\n"
        "- X 없이 △ 1개 이상 → 최대 **우수**\n"
        "- 전부 ○ → 상한 없음"
    )

q_text = st.text_area(
    "📝 문제",
    "그림과 같이 짐을 실은 수레를 말이 350N의 힘으로 끌고, 사람이 150N의 힘으로 밀었다. 사람이 같은 크기의 힘으로 수레를 반대 방향으로 잡아당겼을 때 수레에 작용하는 합력의 방향과 크기를 풀이 과정과 함께 서술하시오.",
    height=200,
)

col1, col2 = st.columns(2)

with col1:
    cond_text = st.text_area(
        "✅ 조건",
        "조건 1: 계산 과정 포함\n조건 2: 크기(단위 포함)와 방향 명시\n조건 3: '합력' 또는 '알짜힘' 용어 사용",
        height=130,
    )

with col2:
    key_text = st.text_area(
        "🔑 필수 키워드",
        "합력, 알짜힘, 200N, 오른쪽(앞방향), 뺄셈식",
        height=130,
    )

ref_text = st.text_area(
    "📚 모범 답안",
    "말이 끄는 힘과 사람이 당기는 힘의 방향이 반대이므로, 합력은 큰 힘에서 작은 힘을 뺀 값이다. 계산식은 350N - 150N = 200N이고, 합력의 크기는 200N이며 방향은 말의 이동 방향(앞방향)이다.",
    height=160,
)

ans_text = st.text_area(
    "✍️ 학생 답안",
    placeholder="학생 답안을 입력하세요.",
    height=160,
)

if st.button("🚀 평가 실행", use_container_width=True):
    answer_error = validate_student_answer(ans_text)
    if answer_error:
        st.warning(answer_error)
        st.stop()

    if not api_key:
        st.error("Gemini API Key를 입력해 주세요.")
        st.stop()

    with st.status("🔍 답안을 분석하고 있습니다...", expanded=True) as status:
        try:
            inputs = {
                "question": q_text,
                "conditions": cond_text,
                "reference": ref_text,
                "keywords": key_text,
                "student_answer": ans_text,
                "api_key": api_key,
            }

            final_state = analysis_app.invoke(inputs)
            report: AnalysisReport = final_state["analysis_result"]

            final_score = calculate_final_score(report)
            raw_level = get_level_from_score(final_score)
            capped_level = cap_level_by_conditions(raw_level, report.individual_conditions)
            final_grade_label = FINAL_GRADE_LABELS[capped_level]

            status.update(label="✅ 분석 완료", state="complete", expanded=False)

        except Exception as e:
            error_msg = str(e)

            status.update(label="❌ 분석 실패", state="error", expanded=True)
            st.error(f"🚨 실제 에러 내용: {error_msg}")

            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                st.error("현재 AI 서버 요청이 많아 분석이 지연되고 있어요. 잠시 후 다시 시도해 주세요.")
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                st.error("API 사용 한도에 도달했어요. 잠시 후 다시 시도하거나 사용량을 확인해 주세요.")
            elif "401" in error_msg or "403" in error_msg:
                st.error("API 인증에 문제가 있어요. API Key 설정을 다시 확인해 주세요.")
            elif "timeout" in error_msg.lower():
                st.error("응답 시간이 초과되었어요. 잠시 후 다시 시도해 주세요.")
            else:
                st.error("분석 중 일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요.")
            st.stop()

        st.success("🎉 평가가 완료되었습니다.")

    st.divider()
    st.subheader("📊 종합 결과")

    levels = ["노력 요함", "보통", "우수", "매우 우수"]
    colors = ["#ff6b6b", "#ffcc5c", "#88d8b0", "#289c46"]
    ranges = [50, 25, 15, 10]

    fig = go.Figure()

    for lvl, color, rng in zip(levels, colors, ranges):
        fig.add_trace(go.Bar(
            y=["등급"],
            x=[rng],
            name=lvl,
            orientation="h",
            marker=dict(color=color, line=dict(color="white", width=1)),
            hoverinfo="name",
            text=[lvl],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=14, family="NanumGothic, sans-serif")
        ))

    fig.update_layout(
        barmode="stack",
        height=92,
        margin=dict(l=0, r=0, t=26, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.add_trace(go.Scatter(
        x=[final_score],
        y=["등급"],
        mode="markers+text",
        marker=dict(color="#111827", size=16, symbol="triangle-down"),
        text=["<b>나의 수준</b>"],
        textposition="top center",
        textfont=dict(color="#111827", size=14, family="NanumGothic, sans-serif")
    ))

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 수정 위치 B: 원점수/최종 결과 카드 높이 통일
    # --- 1. 원점수 / 최종 결과 카드 (이 부분을 통째로 교체!) ---
    score_cards_html = f"""
<div class="score-card-wrap">
    <div class="score-card">
        <div class="score-label">원점수 등급 ⓘ</div>
        <div class="score-value">{raw_level}</div>
        <div style="font-size:0.85rem; color:#9CA3AF; margin-top:4px;">가중치 합산 전 등급</div>
    </div>
    <div class="score-card">
        <div class="score-label">최종 결과 등급 ⓘ</div>
        <div class="score-value">{capped_level}</div>
        <div class="score-sub">↑ {final_score}점 기반</div>
    </div>
</div>
"""
    st.markdown(score_cards_html, unsafe_allow_html=True)

    # --- 2. AI 코치 총평 카드 (이미지 깨짐 해결 버전) ---
    total_card_html = f"""
<div class="total-card">
    <div class="total-card-badge">AI 코치 총평</div>
    <div style="font-size:1.25rem; font-weight:800; color:#1E293B; margin-bottom:12px; line-height:1.4;">
        💌 {report.encouragement}
    </div>
    <div style="margin-bottom:16px;">
        <span class="total-rank">최종 랭크 · {final_grade_label}</span>
    </div>
    <div style="font-size:1.05rem; line-height:1.75; color:#374151; word-break:keep-all;">
        {report.overall_summary}
    </div>
</div>
"""
    st.markdown(total_card_html, unsafe_allow_html=True)

    st.subheader("🔍 항목별 정밀 분석")

    condition_summary = get_condition_summary(report.individual_conditions)

    table_html = dedent(f"""
    <table style="
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        border:1px solid #E5E7EB;
        border-radius:14px;
        overflow:hidden;
        margin-bottom:24px;
        font-size:15px;
        box-shadow:0 6px 16px rgba(15,23,42,0.04);
        background:#FFFFFF;
    ">
        <tr style="background:#F8FAFC; text-align:left;">
            <th style="padding:14px; border-bottom:1px solid #E5E7EB; width:20%;">평가 항목</th>
            <th style="padding:14px; border-bottom:1px solid #E5E7EB; width:20%; text-align:center;">수준 (별점)</th>
            <th style="padding:14px; border-bottom:1px solid #E5E7EB; width:60%;">AI 분석 피드백</th>
        </tr>
        <tr style="background:#FFFFFF;">
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; font-weight:700;">개념 이해 (40%)</td>
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; text-align:center;">{get_star_rating(report.concept_understanding.level)}<br>{report.concept_understanding.level}</td>
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; line-height:1.7;">{report.concept_understanding.feedback}</td>
        </tr>
        <tr style="background:#FCFCFD;">
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; font-weight:700;">논리적 서술 (30%)</td>
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; text-align:center;">{get_star_rating(report.logical_writing.level)}<br>{report.logical_writing.level}</td>
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; line-height:1.7;">{report.logical_writing.feedback}</td>
        </tr>
        <tr style="background:#FFFFFF;">
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; font-weight:700;">용어 사용 (30%)</td>
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; text-align:center;">{get_star_rating(report.term_usage.level)}<br>{report.term_usage.level}</td>
            <td style="padding:14px; border-bottom:1px solid #F1F5F9; line-height:1.7;">{report.term_usage.feedback}</td>
        </tr>
        <tr style="background:#FCFCFD;">
            <td style="padding:14px; font-weight:700;">조건 충족 (등급 상한)</td>
            <td style="padding:14px; text-align:center;">{get_star_rating(condition_summary["level"])}<br>{condition_summary["level"]}</td>
            <td style="padding:14px; line-height:1.7;">{condition_summary["feedback"]}</td>
        </tr>
    </table>
    """)
    st.markdown(table_html, unsafe_allow_html=True)

    st.subheader("🚩 개별 조건 충족 현황")

    # 수정 위치 D: 조건 3개 모두 반복 출력
    for cond in report.individual_conditions:
        bg_color = "#e6fffa" if cond.status == "○" else "#fffaf0" if cond.status == "△" else "#fff5f5"
        border_color = "#38a169" if cond.status == "○" else "#d69e2e" if cond.status == "△" else "#e53e3e"
        label_text = "충족" if cond.status == "○" else "부분 충족" if cond.status == "△" else "미충족"

        box_html = dedent(f"""
        <div style="
            background-color:{bg_color};
            padding:16px 18px;
            border-radius:14px;
            border:1px solid rgba(15,23,42,0.06);
            border-left:6px solid {border_color};
            margin-bottom:14px;
            box-shadow:0 4px 12px rgba(15,23,42,0.03);
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:12px;
                margin-bottom:8px;
            ">
                <div style="font-weight:800; font-size:16px; color:#111827;">
                    {cond.name}
                </div>
                <div style="
                    padding:4px 10px;
                    border-radius:999px;
                    background:#ffffff;
                    font-size:13px;
                    font-weight:700;
                    color:{border_color};
                    border:1px solid {border_color};
                ">
                    {label_text}
                </div>
            </div>
            <div style="font-size:14.5px; color:#374151; line-height:1.65;">
                {cond.reason}
            </div>
        </div>
        """)
        st.markdown(box_html, unsafe_allow_html=True)

    with st.expander("🧮 상세 계산 근거 확인하기"):
        st.markdown(
            f"""
- **개념 이해:** {report.concept_understanding.score}점 × 0.4 = {report.concept_understanding.score * 0.4:.1f}
- **논리적 서술:** {report.logical_writing.score}점 × 0.3 = {report.logical_writing.score * 0.3:.1f}
- **용어 사용:** {report.term_usage.score}점 × 0.3 = {report.term_usage.score * 0.3:.1f}

---
- **가중 합산 점수:** {final_score}점  
- **조건 상한 정책 적용:** {raw_level} → **{capped_level}**
"""
        )
