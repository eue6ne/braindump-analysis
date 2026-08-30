import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_visualization import (
    set_universal_font,
    is_multi_select,
    run_grouped_subplot_eda
)

st.set_page_config(page_title = "EDA 시각화", page_icon = "📊", layout = "wide")

st.title("📊 탐색적 데이터 분석 (EDA)")
st.markdown("---")

TEXT_COLS = ["브레인 덤프", "오늘의 요약"]

if st.session_state.get("cleaned_df") is None:
    st.warning("먼저 데이터 수집 및 전처리를 완료해주세요.")
    st.stop()

df = st.session_state["cleaned_df"].copy()
set_universal_font()

st.success(f"✅ 전처리된 데이터 로드 완료 — {len(df)}행 · 스케일링: {st.session_state.get('clean_suffix', '')}")

if st.button("📊 EDA 시각화 실행", use_container_width = True, type = "primary"):

    target_col = "감정지수"
    if target_col not in df.columns:
        st.error(f"'{target_col}' 컬럼이 없습니다.")
        st.stop()

    numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    categorical_cols = [
        c for c in df.select_dtypes(exclude = [np.number]).columns
        if c != target_col and c not in TEXT_COLS and not is_multi_select(df[c])
    ]
    viz_cat_cols = [c for c in categorical_cols if (df[c].nunique() / len(df[c].dropna())) <= 0.5]

    # 상관관계 히트맵
    st.markdown("## 🔥 수치형 변수 간 상관관계 히트맵")
    st.caption("수치형 변수들 간의 선형 관계 강도를 -1 ~ 1 사이의 값으로 표현합니다.")
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize = (12, 10))
        sns.heatmap(df[numeric_cols].corr(), annot = True, cmap = "coolwarm",
                    fmt = ".2f", linewidths = 0.5, annot_kws = {"size": 8}, ax = ax)
        ax.set_title("수치형 변수 간 상관관계 히트맵", fontsize = 14, weight = "bold", pad = 20)
        plt.xticks(rotation = 45, ha = "right", fontsize = 9)
        plt.yticks(rotation = 0, fontsize = 9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 분포 히스토그램
    st.markdown("## 📈 수치형 변수 분포 히스토그램")
    st.caption("각 수치형 변수의 값 분포와 KDE 곡선을 함께 표시합니다.")
    num_count = len(numeric_cols)
    if num_count > 0:
        ncols = min(3, num_count)
        nrows = math.ceil(num_count / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
        axes = axes.flatten() if num_count > 1 else [axes]
        i = -1
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde = True, ax = axes[i], color = "teal", bins = 10)
            axes[i].set_title(f"[{col}] 분포", fontsize = 11)
            axes[i].set_ylabel("")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle("수치형 변수별 분포 히스토그램", fontsize = 14, weight = "bold", y = 1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 범주형 변수별 박스플롯
    if viz_cat_cols:
        st.markdown("## 📦 범주형 변수별 감정지수 박스플롯")
        st.caption("범주형 변수의 항목별로 감정지수 분포를 비교합니다.")
        cat_count = len(viz_cat_cols)
        ncols = min(3, cat_count)
        nrows = math.ceil(cat_count / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
        axes = axes.flatten() if cat_count > 1 else [axes]
        i = -1
        for i, col in enumerate(viz_cat_cols):
            sns.boxplot(data = df, x = col, y = target_col, ax = axes[i], palette = "Set3")
            sns.stripplot(data = df, x = col, y = target_col, ax = axes[i],
                          color = "black", alpha = 0.3, jitter = 0.1)
            axes[i].set_title(f"[{col}]별 {target_col} 분포", fontsize = 11)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle("범주형 변수별 감정지수 비교 박스플롯", fontsize = 14, weight = "bold", y = 1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("---")

    # 감정지수 추세선
    st.markdown("## 📉 감정지수 추세선")
    st.caption("시간 흐름에 따른 감정지수 변화와 선형 회귀 추세선을 함께 표시합니다.")
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(df.index, df[target_col], marker = "o", color = "lightgray",
            markersize = 5, linewidth = 1.2, label = f"실제 {target_col}")
    x_ord = np.array([date.toordinal() for date in df.index])
    slope, intercept = np.polyfit(x_ord, df[target_col].values, 1)
    trend_color = "crimson" if slope >= 0 else "dodgerblue"
    ax.plot(df.index, slope * x_ord + intercept, color = trend_color,
            linewidth = 2.5, label = f"추세선 (기울기: {slope:.4f})")
    ax.set_title(f"시간 흐름에 따른 {target_col} 추세선 분석", fontsize = 12, pad = 10)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.legend(fontsize = 10)
    plt.xticks(rotation = 35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 다중선택 항목별 막대그래프
    encoded_groups = {}
    for col in df.select_dtypes(include = [np.number]).columns:
        if col == target_col or "_" not in col:
            continue
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            group, item = parts
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                if group not in encoded_groups:
                    encoded_groups[group] = []
                encoded_groups[group].append((item, col))

    for group, items in encoded_groups.items():
        if len(items) < 2:
            continue
        st.markdown(f"## 🎯 [{group}] 항목별 평균 감정지수")
        item_labels = [item for item, _ in items]
        item_means = [df[col].mul(df[target_col]).sum() / df[col].sum()
                      if df[col].sum() > 0 else 0 for _, col in items]
        fig, ax = plt.subplots(figsize = (max(6, len(items) * 1.2), 4))
        ax.bar(item_labels, item_means, color = "steelblue", alpha = 0.8, edgecolor = "white")
        ax.axhline(y = df[target_col].mean(), color = "crimson", linestyle = "--",
                   linewidth = 1.5, label = f"전체 평균 ({df[target_col].mean():.2f})")
        ax.set_title(f"[{group}] 항목별 평균 {target_col}", fontsize = 12, weight = "bold")
        ax.set_ylabel(target_col)
        ax.set_ylim(0, 10)
        ax.legend(fontsize = 9)
        plt.xticks(rotation = 30, ha = "right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("---")

    # 요일별 박스플롯
    st.markdown("## 📅 요일별 감정지수 분포")
    st.caption("요일별 감정지수 분포를 비교합니다.")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = ["월", "화", "수", "목", "금", "토", "일"]
    day_map = dict(zip(day_order, day_label))
    df_day = df.copy()
    df_day["요일"] = df_day.index.day_name().map(day_map)
    existing_days = [d for d in day_label if d in df_day["요일"].values]
    if len(existing_days) >= 2:
        fig, ax = plt.subplots(figsize = (9, 4))
        sns.boxplot(data = df_day, x = "요일", y = target_col,
                    order = existing_days, palette = "pastel", ax = ax)
        sns.stripplot(data = df_day, x = "요일", y = target_col,
                      order = existing_days, color = "black", alpha = 0.4, jitter = 0.1, ax = ax)
        ax.set_title(f"요일별 {target_col} 분포", fontsize = 12, weight = "bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✅ EDA 시각화 완료!")