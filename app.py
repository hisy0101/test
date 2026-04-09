import streamlit as st
from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import plotly.graph_objects as go

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
    total = len(conditions)
    if total == 0:
        return {
            "level": "보통",
            "feedback": "확인할 조건 정보가 없어서 조건 충족 여부를 판단하기 어려워. 문항에 제시된 요구 조건을 먼저 정리해 두면 더 정확하게 볼 수 있어."
        }

    complete = sum(1 for c in conditions if c.status == "○")
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
# 5. 분석 노드
# =========================
def analysis_node(state: GradingState):
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=state["api_key"],
        temperature=0.1,  # 평가 일관성을 높이기 위해 낮춤
    )

    structured_llm = llm.with_structured_output(AnalysisReport)

    system_message = """
당신은 중학생 서술형 답안을 채점하는 '지적인 선배 코치'입니다. 다음 원칙에 따라 분석 리포트를 작성하세요.

[말투 및 문체]
- 친절하고 명확한 선배 말투 사용 (교사 말투/유치한 표현 금지).
- 국립국어원 표준 맞춤법 및 띄어쓰기 원칙 엄격 준수 (보조 용언 띄어쓰기 필수).
- 이모지 사용 지양. 문장은 짧고 간결하게 작성.

[평가 및 피드백 원칙]
1. 개념 이해: 핵심 원리 파악 여부 평가.
2. 논리적 서술: '이유-과정-결론'의 흐름 평가. 결과만 있으면 감점.
3. 용어 사용: 교과 용어의 정확성 평가.
4. 조건 충족: 제시된 [미션 조건] 반영 여부를 ○, △, X로 판정.

[피드백 작성 공식 (2~4문장)]
- 1문장: 잘한 점 또는 현재 상태 칭찬.
- 2문장: 부족한 점과 그 이유 설명.
- 3~4문장: "이렇게 바꾸면 더 좋아"와 같은 구체적 예시나 실전 팁 제공.

[점수 및 등급 매핑]
- 90~100: 매우 우수 / 75~89: 우수 / 50~74: 보통 / 0~49: 노력 요함
- 항목별 점수(score)와 수준(level)은 반드시 위 기준에 맞춰 일치시킬 것.

[출력 데이터 구성]
- final_score 및 최종 등급은 직접 생성하지 말 것 (로직에서 처리됨).
- 각 항목(개념, 논리, 용어)의 score, level, feedback을 작성.
- individual_conditions: 각 조건의 status(○, △, X)와 쉬운 설명(reason) 작성.
- overall_summary: 강점/약점/개선 포인트를 포함한 총평 (1~2문단).
- encouragement: 성의 있고 따뜻한 한 줄 격려.

학생이 피드백만 읽고도 스스로 답안을 고쳐 쓸 수 있도록 구체적인 '행동 가이드'를 주는 데 집중하세요.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", """
[문제]
{question}

[미션 조건]
{conditions}

[필수 키워드]
{keywords}

[모범 답안]
{reference}

[학생 답안]
{student_answer}
"""),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "question": state["question"],
            "conditions": state["conditions"],
            "keywords": state["keywords"],
            "reference": state["reference"],
            "student_answer": state["student_answer"],
        }
    )

    return {"analysis_result": result}


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
st.set_page_config(page_title="AI 서술형 평가 코치", layout="wide")

st.markdown("""
<style>
/* ===== 메인 본문 시작 높이 조정 ===== */
section.main > div {
    padding-top: 0.2rem !important;
}

section.main div.block-container {
    padding-top: 0.35rem !important;
    padding-bottom: 0.5rem !important;
}

/* ===== 본문 기본 ===== */
html, body, [class*="css"] {
    font-size: 100% !important;
}

p, li, div {
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
}

/* 혹시 남아 있는 Streamlit 기본 제목 스타일 영향 최소화 */
div[data-testid="stHeading"] h1,
div[data-testid="stHeadingWithActionElements"] h1 {
    font-size: inherit !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}

/* ===== 캡션 ===== */
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

/* ===== 섹션 제목 ===== */
h2 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 1.28rem !important;
    font-weight: 700 !important;
}

/* ===== 사이드바 ===== */
section[data-testid="stSidebar"] * {
    font-size: 1.02rem !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
}

/* ===== 입력 라벨 ===== */
label[data-testid="stWidgetLabel"] p {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    line-height: 1.45 !important;
}

/* ===== 입력창 내부 ===== */
textarea, input {
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

/* ===== 버튼 ===== */
button[kind="primary"] p,
button p {
    font-size: 1.06rem !important;
    font-weight: 700 !important;
}

/* ===== metric ===== */
div[data-testid="stMetricLabel"] {
    font-size: 1.02rem !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* ===== 알림 박스 ===== */
[data-testid="stAlertContainer"] p {
    font-size: 1rem !important;
    line-height: 1.55 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
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
""", unsafe_allow_html=True)
st.caption("LLM은 항목별 판단과 피드백을 생성하고, 최종 점수와 등급은 Python 로직으로 계산합니다.")
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

            st.error(f"🚨 실제 에러 내용: {error_msg}") # 진짜 범인의 이름표를 보여줘!

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

    # --- ✅ 수정 2: 기존 metric 코드를 지우고, 이 차트 로직으로 교체해! ---
    
    # 1. 차트 데이터 설정 (등급별 점수 구간과 색상)
    levels = ["노력 요함", "보통", "우수", "매우 우수"]
    colors = ["#ff6b6b", "#ffcc5c", "#88d8b0", "#289c46"]  # 빨-노-초 (학생 마음에 쏙 드는 색!)
    ranges = [50, 25, 15, 10]  # 각 등급의 구간 길이 (0~49, 50~74, 75~89, 90~100)
    
    # 2. 멋진 Plotly 가로 막대 차트 생성
    fig = go.Figure()
    
    for lvl, color, rng in zip(levels, colors, ranges):
        fig.add_trace(go.Bar(
            y=['등급'], 
            x=[rng], 
            name=lvl, 
            orientation='h', 
            marker=dict(color=color, line=dict(color='white', width=1)),
            hoverinfo='name',
            text=[lvl] if lvl == capped_level else [""], # 현재 등급만 글씨 표시
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white', size=14, family='NanumGothic, sans-serif')
        ))

    # 3. 차트 레이아웃 꾸미기 (UI 깔끔하게!)
    fig.update_layout(
        barmode='stack', # 누적 막대
        height=100,      # 높이는 슬림하게
        margin=dict(l=0, r=0, t=20, b=0), # 여백 최소화
        showlegend=False, # 범례는 숨김 (직관성 위해)
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]), # 축 숨김
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), # 축 숨김
        paper_bgcolor='rgba(0,0,0,0)', # 배경 투명
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # 현재 점수 위치에 바늘/점 표시 (학생의 위치 확인!)
    fig.add_trace(go.Scatter(
        x=[final_score], 
        y=['등급'], 
        mode='markers+text',
        marker=dict(color='black', size=15, symbol='triangle-up'),
        text=[f'<b>{final_score}점</b>'],
        textposition='top center',
        textfont=dict(color='black', size=16, family='NanumGothic, sans-serif')
    ))

    # 4. Streamlit에 차트 그리기
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # (기존 metric 중 중요한 것만 아래 소형으로 배치)
    col1, col2 = st.columns([1, 1])
    col1.metric("원점수 등급", raw_level, help="조건 충족 전 점수 기반 등급")
    col2.metric("최종 결과 등급", capped_level, f"{final_score}점 기반", help="조건 충족 여부를 반영한 최종 등급")

    with st.container(border=True):
        st.markdown(f"### 💌 {report.encouragement}")
        st.markdown(f"**최종 랭크:** `{final_grade_label}`")
        st.write(report.overall_summary)

    # --- ✅ 수정 1: 항목별 정밀 분석 (삼분할 HTML 표) ---
    st.subheader("🔍 항목별 정밀 분석")

    condition_summary = get_condition_summary(report.individual_conditions)

    main_table_html = f"""

    <table style="width:100%; border-collapse: collapse; border: 1px solid #ddd; margin-bottom: 20px; font-size: 15px;">
        <tr style="background-color: #f8f9fa; text-align: left;">
            <th style="padding: 12px; border: 1px solid #ddd; width: 20%;">평가 항목</th>
            <th style="padding: 12px; border: 1px solid #ddd; width: 20%; text-align: center;">수준 (별점)</th>
            <th style="padding: 12px; border: 1px solid #ddd; width: 60%;">선배의 족집게 피드백 & 꿀팁 🎯</th>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">개념 이해 (40%)</td>
            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{get_star_rating(report.concept_understanding.level)}<br>{report.concept_understanding.level}</td>
            <td style="padding: 12px; border: 1px solid #ddd; line-height: 1.6;">{report.concept_understanding.feedback}</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">논리적 서술 (30%)</td>
            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{get_star_rating(report.logical_writing.level)}<br>{report.logical_writing.level}</td>
            <td style="padding: 12px; border: 1px solid #ddd; line-height: 1.6;">{report.logical_writing.feedback}</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">용어 사용 (30%)</td>
            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{get_star_rating(report.term_usage.level)}<br>{report.term_usage.level}</td>
            <td style="padding: 12px; border: 1px solid #ddd; line-height: 1.6;">{report.term_usage.feedback}</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">조건 충족 (등급 상한)</td>
            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{get_star_rating(condition_summary["level"])}<br>{condition_summary["level"]}</td>
            <td style="padding: 12px; border: 1px solid #ddd; line-height: 1.6;">{condition_summary["feedback"]}</td>
        </tr>
    </table>
    """
    st.markdown(main_table_html, unsafe_allow_html=True)

    # --- ✅ 수정 2: 조건 충족 현황 (기호 후 줄바꿈 박스 스타일) ---
    st.subheader("🚩 개별 조건 충족 현황")

    for cond in report.individual_conditions:
        # 상태에 따른 배경색 지정
        bg_color = "#e6fffa" if cond.status == "○" else "#fffaf0" if cond.status == "△" else "#fff5f5"
        border_color = "#38a169" if cond.status == "○" else "#d69e2e" if cond.status == "△" else "#e53e3e"
        
        box_html = f"""
        <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; border-left: 5px solid {border_color}; margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 16px; margin-bottom: 4px;">{cond.name} | <span style="font-size: 20px;">{cond.status}</span></div>
            <br>
            <div style="font-size: 15px; color: #2d3748; line-height: 1.6;">{cond.reason}</div>
        </div>
        """
        st.markdown(box_html, unsafe_allow_html=True)

    # --- ✅ 수정 3: 계산 근거 ---
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
