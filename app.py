import streamlit as st

st.set_page_config(
    page_title = "브레인 덤프 감정 분석",
    page_icon = "📔",
    layout = "wide"
)

# 세션 상태 초기화
if "notion_token" not in st.session_state:
    st.session_state["notion_token"] = ""
if "database_id" not in st.session_state:
    st.session_state["database_id"] = ""
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "clean_suffix" not in st.session_state:
    st.session_state["clean_suffix"] = ""

st.title("📔 브레인 덤프 데이터를 활용한 개인 로그 데이터 분석")
st.markdown("---")

st.markdown("""
매일 기록한 브레인 덤프 데이터를 수집하고, 전처리 및 통계 분석을 수행하는 개인 데이터 분석 프로젝트입니다.

| 분석 | 목적 |
|------|------|
| EDA 시각화 | 변수 분포 및 상관관계 탐색 |
| 요인분석 (PCA) | 변수 구조 파악 |
| 회귀분석 | 감정지수 예측 모델 |
| 클러스터링 | 하루 패턴 유형화 |
| 시계열 분석 | 감정 추세 및 주기성 |
| 텍스트 마이닝 | 키워드 기반 감정 탐색 |
| NLP 감정 분석 | 텍스트 감정 점수 산출 (보조) |
""")

st.markdown("---")
st.markdown("## 🔗 Notion 연결")
st.info("분석을 시작하려면 Notion API 토큰과 데이터베이스 ID를 입력하세요. 입력한 정보는 이 세션에서만 사용되며 저장되지 않습니다.")

with st.form("notion_form"):
    token_input = st.text_input(
        "Notion API 토큰",
        type = "password",
        value = st.session_state["notion_token"],
        placeholder = "ntn_xxxxxxxxxxxx"
    )
    db_input = st.text_input(
        "데이터베이스 ID",
        value = st.session_state["database_id"],
        placeholder = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    submitted = st.form_submit_button("연결 저장")

    if submitted:
        if token_input and db_input:
            st.session_state["notion_token"] = token_input
            st.session_state["database_id"] = db_input
            st.success("✅ Notion 연결 정보가 저장되었습니다. 왼쪽 사이드바에서 데이터 수집 및 전처리를 먼저 진행하세요.")
        else:
            st.error("토큰과 데이터베이스 ID를 모두 입력해주세요.")

if st.session_state["notion_token"] and st.session_state["database_id"]:
    st.success(f"✅ 연결 정보 저장됨 — 사이드바에서 분석을 시작하세요.")

st.markdown("---")
st.caption("Personal Data Analysis Project · 2026")