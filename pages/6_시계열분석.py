import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_visualization import set_universal_font
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title = "시계열 분석", page_icon = "📈", layout = "wide")

st.title("📈 시계열 분석")
st.markdown("---")

if st.session_state.get("cleaned_df") is None:
    st.warning("먼저 데이터 수집 및 전처리를 완료해주세요.")
    st.stop()

df = st.session_state["cleaned_df"].copy()
set_universal_font()

st.success(f"✅ 전처리된 데이터 로드 완료 — {len(df)}행 · 스케일링: {st.session_state.get('clean_suffix', '')}")
st.info("💡 시계열 분석은 원래 스케일 그대로 봐야 해석이 직관적이므로 none 스케일링 파일을 권장합니다.")

st.markdown("""
감정지수의 시간적 패턴, 추세, 주기성을 탐색하는 분석입니다.
이동평균, 시계열 분해, 자기상관 분석, 요일별 패턴을 통해 감정의 흐름을 파악합니다.
""")

col1, col2 = st.columns(2)
with col1:
    period = st.number_input("시계열 분해 주기 (일)", min_value = 2, max_value = 30, value = 7,
                              help = "주간 패턴 기준 7일, 월간 패턴은 30")
with col2:
    lags = st.number_input("ACF / PACF 최대 lag 수", min_value = 5, max_value = 60, value = 21)

if st.button("📈 시계열 분석 실행", use_container_width = True, type = "primary"):

    target_col = "감정지수"
    y = df[target_col].asfreq("D")

    # ADF 정상성 검정
    result = adfuller(y.dropna())
    p_value = result[1]
    st.markdown("---")
    st.markdown("## 🔬 ADF 정상성 검정")
    col1, col2 = st.columns(2)
    col1.metric("ADF 통계량", f"{result[0]:.4f}")
    col2.metric("p-value", f"{p_value:.4f}")
    if p_value < 0.05:
        st.success("✅ 정상 시계열 (추세 없음)")
    else:
        st.warning("⚠️ 비정상 시계열 (추세 또는 계절성 존재 가능)")

    st.markdown("---")

    # 추이 + 이동평균
    st.markdown("## 📉 감정지수 추이 및 이동평균")
    st.caption("7일 이동평균(파란 실선)과 14일 이동평균(빨간 점선)으로 전반적인 감정 흐름을 파악합니다.")
    ma7 = y.rolling(window = 7, center = True).mean()
    ma14 = y.rolling(window = 14, center = True).mean()

    fig, ax = plt.subplots(figsize = (12, 5))
    ax.plot(y.index, y.values, color = "lightgray", linewidth = 1.0, alpha = 0.8, label = "실제 감정지수")
    ax.plot(ma7.index, ma7.values, color = "steelblue", linewidth = 2.0, label = "7일 이동평균")
    ax.plot(ma14.index, ma14.values, color = "crimson", linewidth = 2.0, linestyle = "--", label = "14일 이동평균")
    ax.set_title("감정지수 추이 및 이동평균", fontsize = 12, weight = "bold")
    ax.set_ylabel(target_col)
    ax.legend(fontsize = 10)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation = 35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 시계열 분해
    st.markdown(f"## 🔬 시계열 분해 (주기 = {period}일)")
    st.caption("원본 시계열을 추세, 계절성(주기적 패턴), 잔차로 분리합니다.")
    y_filled = y.interpolate(method = "linear")
    decomp = seasonal_decompose(y_filled, model = "additive", period = period)

    fig, axes = plt.subplots(4, 1, figsize = (12, 10), sharex = True)
    axes[0].plot(y.index, y.values, color = "steelblue", linewidth = 1.2)
    axes[0].set_ylabel("원본")
    axes[0].set_title(f"시계열 분해 (주기 = {period}일 / 가법 모형)", fontsize = 13, weight = "bold")
    axes[1].plot(decomp.trend.index, decomp.trend.values, color = "crimson", linewidth = 1.5)
    axes[1].axhline(y = y.mean(), color = "gray", linestyle = ":", linewidth = 1.0)
    axes[1].set_ylabel("추세")
    axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color = "darkorange", linewidth = 1.2)
    axes[2].axhline(y = 0, color = "gray", linestyle = ":", linewidth = 1.0)
    axes[2].set_ylabel("계절성")
    axes[3].plot(decomp.resid.index, decomp.resid.values, color = "gray", linewidth = 1.0, alpha = 0.8)
    axes[3].axhline(y = 0, color = "black", linestyle = "--", linewidth = 0.8)
    axes[3].set_ylabel("잔차")
    for ax in axes:
        ax.grid(True, linestyle = ":", alpha = 0.4)
    plt.xticks(rotation = 35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ACF / PACF
    st.markdown("## 🔗 자기상관 분석 (ACF / PACF)")
    st.caption("파란 음영 밖으로 튀어나온 막대가 통계적으로 유의미한 자기상관을 나타냅니다.")
    fig, axes = plt.subplots(2, 1, figsize = (12, 7))
    plot_acf(y.dropna(), lags = lags, ax = axes[0], alpha = 0.05, color = "steelblue")
    axes[0].set_title("ACF (자기상관함수)", fontsize = 12, weight = "bold")
    plot_pacf(y.dropna(), lags = lags, ax = axes[1], alpha = 0.05, color = "darkorange", method = "ywm")
    axes[1].set_title("PACF (편자기상관함수)", fontsize = 12, weight = "bold")
    plt.suptitle("자기상관 분석", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 요일별 패턴
    st.markdown("## 📅 주간 감정 패턴 분석")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = ["월", "화", "수", "목", "금", "토", "일"]
    day_map = dict(zip(day_order, day_label))
    df_day = df[[target_col]].copy()
    df_day["요일"] = df_day.index.day_name().map(day_map)
    existing = [d for d in day_label if d in df_day["요일"].values]

    fig, axes = plt.subplots(1, 2, figsize = (13, 5))
    means = df_day.groupby("요일")[target_col].mean().reindex(existing)
    overall_mean = df_day[target_col].mean()
    bar_colors = ["crimson" if v >= overall_mean else "steelblue" for v in means.values]
    axes[0].bar(means.index, means.values, color = bar_colors, alpha = 0.8, edgecolor = "white")
    axes[0].axhline(y = overall_mean, color = "black", linestyle = "--", linewidth = 1.2,
                    label = f"전체 평균 ({overall_mean:.1f})")
    axes[0].set_title("요일별 평균 감정지수", fontsize = 11, weight = "bold")
    axes[0].set_ylabel(target_col)
    axes[0].legend(fontsize = 9)
    axes[0].set_ylim(0, 10)
    sns.boxplot(data = df_day, x = "요일", y = target_col, order = existing, palette = "pastel", ax = axes[1])
    sns.stripplot(data = df_day, x = "요일", y = target_col, order = existing,
                  color = "black", alpha = 0.4, jitter = 0.1, ax = axes[1])
    axes[1].axhline(y = overall_mean, color = "crimson", linestyle = "--", linewidth = 1.2)
    axes[1].set_title("요일별 감정지수 분포", fontsize = 11, weight = "bold")
    plt.suptitle("주간 감정 패턴 분석", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 핵심 통계 요약
    st.markdown("---")
    st.markdown("## 📊 핵심 통계 요약")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균 감정지수", f"{y.mean():.2f}")
    col2.metric("표준편차", f"{y.std():.2f}")
    col3.metric("최고점", f"{y.max():.1f} ({y.idxmax().date()})")
    col4.metric("최저점", f"{y.min():.1f} ({y.idxmin().date()})")

    ma7_clean = y.rolling(window = 7, center = True).mean().dropna()
    if len(ma7_clean) >= 2:
        slope = np.polyfit(np.arange(len(ma7_clean)), ma7_clean.values, 1)[0]
        direction = "상승 📈" if slope > 0.01 else "하락 📉" if slope < -0.01 else "보합 ➡️"
        st.info(f"전반적 추세: {direction} (7일 이동평균 기울기: {slope:.4f})")

    st.success("✅ 시계열 분석 완료!")