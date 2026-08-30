import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    handle_drop,
    handle_impute,
    handle_encode_multiselect,
    handle_scaling,
    get_col_types,
    SCALE_EXCLUDE_COLS
)

st.set_page_config(page_title = "데이터 수집 및 전처리", page_icon = "🗃️", layout = "wide")

st.title("🗃️ 데이터 수집 및 전처리")
st.markdown("---")

# ── 1. 데이터 수집 ──────────────────────────────────────────
st.markdown("## 1️⃣ 데이터 수집")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Notion API로 수집")
    if not st.session_state.get("notion_token") or not st.session_state.get("database_id"):
        st.warning("홈 화면에서 Notion 연결 정보를 먼저 입력해주세요.")
    else:
        if st.button("🔄 Notion에서 데이터 수집", use_container_width = True):
            with st.spinner("노션 데이터베이스에서 데이터를 가져오는 중..."):
                try:
                    from notion_client import Client
                    from notion_loader import fetch_notion_data, parse_notion_properties
                    results = fetch_notion_data(st.session_state["database_id"])
                    df = parse_notion_properties(results)
                    st.session_state["raw_df"] = df
                    st.session_state["cleaned_df"] = None
                    st.session_state["outlier_info"] = None
                    st.success(f"✅ {len(df)}행 수집 완료!")
                except Exception as e:
                    st.error(f"오류 발생: {e}")

with col2:
    st.markdown("### 샘플 데이터로 시작")
    st.caption("Notion 연결 없이 샘플 데이터로 분석을 진행합니다.")
    if st.button("📂 샘플 데이터 불러오기", use_container_width = True):
        sample_path = "data/sample_data.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            st.session_state["raw_df"] = df
            st.session_state["cleaned_df"] = None
            st.session_state["outlier_info"] = None
            st.success(f"✅ 샘플 데이터 {len(df)}행 로드 완료!")
        else:
            st.error(f"샘플 데이터 파일을 찾을 수 없습니다: {sample_path}")

if st.session_state.get("raw_df") is not None:
    raw_df = st.session_state["raw_df"]
    st.markdown("#### 📋 수집된 데이터 미리보기")
    st.dataframe(raw_df.head(10), use_container_width = True)
    st.caption(f"총 {len(raw_df)}행 · {len(raw_df.columns)}개 컬럼")

st.markdown("---")

# ── 2. 전처리 옵션 ─────────────────────────────────────────
st.markdown("## 2️⃣ 전처리 옵션 선택")

if st.session_state.get("raw_df") is None:
    st.info("먼저 데이터를 수집하거나 샘플 데이터를 불러오세요.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    mode = st.selectbox(
        "결측치 처리 방식",
        options = ["impute", "drop"],
        help = "impute: KNN 대체 (데이터 적을 때 권장) / drop: 행 제거 (90일+ 권장)"
    )

with col2:
    scaler = st.selectbox(
        "스케일링",
        options = ["none", "standard", "minmax"],
        help = "none: EDA·시계열 / standard: 회귀·요인분석 / minmax: 클러스터링"
    )

st.caption("💡 분석별 권장 스케일링: EDA·시계열 → none / 회귀·요인분석 → standard / 클러스터링 → minmax")

st.markdown("---")

# ── 3. 이상치 탐지 ─────────────────────────────────────────
st.markdown("## 3️⃣ 이상치 탐지 및 처리")

if st.button("🔍 이상치 탐지 실행", use_container_width = True):
    df = st.session_state["raw_df"].copy()

    # 기본 필터링
    if "오늘의 요약" in df.columns:
        df = df.dropna(subset = ["오늘의 요약"])
    if "감정지수" in df.columns:
        df = df.dropna(subset = ["감정지수"])
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"])
        df = df.sort_values("날짜").set_index("날짜")

    # 결측치 처리
    if mode == "drop":
        df = handle_drop(df)
    else:
        df = handle_impute(df)

    # IQR 이상치 탐지
    _, _, numeric_cols, _ = get_col_types(df)
    target_cols = [c for c in numeric_cols if c not in SCALE_EXCLUDE_COLS]

    outlier_info = {}
    for col in target_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_outlier = (df[col] < lower) | (df[col] > upper)
        if is_outlier.sum() > 0:
            outlier_info[col] = {
                "lower": lower,
                "upper": upper,
                "indices": df[is_outlier].index.tolist(),
                "values": df[is_outlier][col].tolist()
            }

    st.session_state["outlier_info"] = outlier_info
    st.session_state["pre_outlier_df"] = df

    if outlier_info:
        st.warning(f"⚠️ {len(outlier_info)}개 컬럼에서 이상치가 탐지되었습니다.")
    else:
        st.success("✅ 탐지된 이상치가 없습니다.")

# 이상치 처리 UI
if st.session_state.get("outlier_info") is not None:
    outlier_info = st.session_state["outlier_info"]

    if outlier_info:
        st.markdown("#### 탐지된 이상치 목록")
        method = st.radio(
            "처리 방식 선택",
            options = ["건너뜀 (원본 유지)", "flag (이상치_플래그 컬럼 추가)", "cap (IQR 경계값으로 대체)"],
            horizontal = True
        )

        selected = {}
        for col, info in outlier_info.items():
            st.markdown(f"**{col}** — 정상 범위: `{info['lower']:.2f} ~ {info['upper']:.2f}`")
            options = [f"{str(idx)[:10]}  |  값: {val}" for idx, val in zip(info["indices"], info["values"])]
            chosen = st.multiselect(
                f"{col} 에서 처리할 항목 선택 (전체 선택 시 모두 처리)",
                options = options,
                default = options,
                key = f"outlier_{col}"
            )
            selected[col] = [info["indices"][options.index(c)] for c in chosen]

        if st.button("✅ 이상치 처리 적용 후 전처리 완료", use_container_width = True, type = "primary"):
            df = st.session_state["pre_outlier_df"].copy()

            if "건너뜀" not in method:
                flag_indices = []
                for col, indices in selected.items():
                    if not indices:
                        continue
                    info = outlier_info[col]
                    if "flag" in method:
                        flag_indices.extend(indices)
                    elif "cap" in method:
                        for idx in indices:
                            original = df.at[idx, col]
                            df.at[idx, col] = max(info["lower"], min(info["upper"], original))
                if "flag" in method and flag_indices:
                    unique_flags = list(dict.fromkeys(flag_indices))
                    df["이상치_플래그"] = df.index.isin(unique_flags).astype(int)

            df = handle_encode_multiselect(df)
            df = handle_scaling(df, scaler)
            st.session_state["cleaned_df"] = df
            st.session_state["clean_suffix"] = f"{mode}_{scaler}"
            st.success("✅ 전처리 완료!")

    else:
        # 이상치 없으면 바로 전처리 완료
        if st.button("✅ 전처리 완료 (이상치 없음)", use_container_width = True, type = "primary"):
            df = st.session_state["pre_outlier_df"].copy()
            df = handle_encode_multiselect(df)
            df = handle_scaling(df, scaler)
            st.session_state["cleaned_df"] = df
            st.session_state["clean_suffix"] = f"{mode}_{scaler}"
            st.success("✅ 전처리 완료!")

st.markdown("---")

# ── 4. 전처리 결과 ─────────────────────────────────────────
if st.session_state.get("cleaned_df") is not None:
    cleaned_df = st.session_state["cleaned_df"]
    st.markdown("## 4️⃣ 전처리 결과")
    st.dataframe(cleaned_df.head(10), use_container_width = True)
    st.caption(f"총 {len(cleaned_df)}행 · {len(cleaned_df.columns)}개 컬럼 · 스케일링: {st.session_state['clean_suffix']}")

    missing = cleaned_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.warning(f"잔여 결측치: {missing.to_dict()}")
    else:
        st.success("결측치 없음")

    csv = cleaned_df.reset_index().to_csv(index = False, encoding = "utf-8-sig")
    st.download_button(
        label = "📥 전처리된 데이터 CSV 다운로드",
        data = csv,
        file_name = f"cleaned_{st.session_state['clean_suffix']}.csv",
        mime = "text/csv"
    )