import warnings
warnings.filterwarnings("ignore")

import os
import platform
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 시각화 이미지 저장 경로
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok = True)

TARGET_COL = "감정지수"

def set_universal_font():
    """운영체제(OS)를 자동으로 감지하여 한글 깨짐이 없도록 폰트를 설정하는 함수"""
    os_name = platform.system()
    if os_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif os_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else: # Linux: sudo apt install fonts-nanum / rm -rf ~/.cache/matplotlib
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

def load_data(file_path):
    """전처리된 CSV 로드 및 시계열 형태로 반환"""
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    if TARGET_COL not in df.columns:
        raise ValueError(f"[오류] '{TARGET_COL}' 컬럼이 없습니다.")

    y = df[TARGET_COL].asfreq("D")  # 일별 주기 명시
    print(f"[데이터 로드] {len(df)}행 / 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df, y

def test_stationarity(y):
    """
    ADF(Augmented Dickey-Fuller) 검정으로 정상성 확인.
    - 정상성(Stationarity): 평균과 분산이 시간에 따라 일정한 상태
    - p-value < 0.05: 정상 시계열 (시간에 따른 추세 없음)
    - p-value >= 0.05: 비정상 시계열 (추세 또는 계절성 존재 가능)
    """
    result = adfuller(y.dropna())
    p_value = result[1]
    print(f"\n[ADF 정상성 검정]")
    print(f"  ADF 통계량: {result[0]:.4f}")
    print(f"  p-value   : {p_value:.4f}")
    if p_value < 0.05:
        print("  → 정상 시계열 (추세 없음)")
    else:
        print("  → 비정상 시계열 (추세 또는 계절성 존재 가능)")
    return p_value

def plot_trend_with_ma(y):
    """
    감정지수 원본 + 이동평균 시각화.
    - 7일 이동평균: 단기 노이즈 제거, 주간 흐름 파악
    - 14일 이동평균: 중기 추세 파악
    """
    ma7  = y.rolling(window = 7, center = True).mean()
    ma14 = y.rolling(window = 14, center = True).mean()

    plt.figure(figsize = (12, 5))
    plt.plot(y.index, y.values, color = "lightgray", linewidth = 1.0, alpha = 0.8, label = "실제 감정지수")
    plt.plot(ma7.index, ma7.values, color = "steelblue", linewidth = 2.0, label = "7일 이동평균")
    plt.plot(ma14.index, ma14.values, color = "crimson", linewidth = 2.0, linestyle = "--", label = "14일 이동평균")
    plt.title("감정지수 추이 및 이동평균\n(이동평균선이 전반적인 감정 흐름을 보여줍니다)", fontsize = 12, weight = "bold")
    plt.ylabel(TARGET_COL)
    plt.legend(fontsize = 10)
    plt.grid(True, linestyle = ":", alpha = 0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation = 35)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/timeseries_trend.png", dpi = 300)
    plt.close()
    print("[추이 + 이동평균] 저장 완료")

def plot_decomposition(y, period = 7):
    """
    시계열 분해: 추세 / 계절성 / 잔차로 분리.
    - period=7: 주간 계절성 기준 (7일 주기)
    - 추세(Trend): 장기적인 감정 방향
    - 계절성(Seasonal): 반복되는 주기적 패턴 (주간 패턴 등)
    - 잔차(Residual): 추세와 계절성으로 설명되지 않는 부분
    """
    # 결측치가 있으면 보간 후 분해
    y_filled = y.interpolate(method = "linear")
    decomp = seasonal_decompose(y_filled, model = "additive", period = period)

    fig, axes = plt.subplots(4, 1, figsize = (12, 10), sharex = True)

    axes[0].plot(y.index, y.values, color = "steelblue", linewidth = 1.2)
    axes[0].set_ylabel("원본")
    axes[0].set_title(f"시계열 분해 (주기 = {period}일 / 가법 모형)", fontsize = 13, weight = "bold")

    axes[1].plot(decomp.trend.index, decomp.trend.values, color = "crimson", linewidth = 1.5)
    axes[1].set_ylabel("추세")
    axes[1].axhline(y = y.mean(), color = "gray", linestyle = ":", linewidth = 1.0)

    axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color = "darkorange", linewidth = 1.2)
    axes[2].set_ylabel("계절성")
    axes[2].axhline(y = 0, color = "gray", linestyle = ":", linewidth = 1.0)

    axes[3].plot(decomp.resid.index, decomp.resid.values, color = "gray", linewidth = 1.0, alpha = 0.8)
    axes[3].axhline(y = 0, color = "black", linestyle = "--", linewidth = 0.8)
    axes[3].set_ylabel("잔차")

    for ax in axes:
        ax.grid(True, linestyle = ":", alpha = 0.4)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation = 35)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/timeseries_decomposition.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[시계열 분해] 저장 완료 (주기 = {period}일)")

def plot_acf_pacf(y, lags = 21):
    """
    ACF / PACF 시각화.
    - ACF(자기상관함수): lag k일 전 값과의 상관관계
    - PACF(편자기상관함수): 중간 lag의 영향을 제거한 순수 상관관계
    - 파란 음영 밖으로 튀어나온 막대: 통계적으로 유의미한 자기상관
    """
    fig, axes = plt.subplots(2, 1, figsize = (12, 7))

    plot_acf(y.dropna(), lags = lags, ax = axes[0], alpha = 0.05, color = "steelblue")
    axes[0].set_title("ACF (자기상관함수)\n(음영 밖 막대: 통계적으로 유의미한 자기상관)", fontsize = 12, weight = "bold")
    axes[0].set_xlabel("Lag (일)")

    plot_pacf(y.dropna(), lags = lags, ax = axes[1], alpha = 0.05, color = "darkorange", method = "ywm")
    axes[1].set_title("PACF (편자기상관함수)\n(직접적인 자기상관만 표시)", fontsize = 12, weight = "bold")
    axes[1].set_xlabel("Lag (일)")

    plt.suptitle("자기상관 분석 — 감정지수가 며칠 전 값과 얼마나 연관되는지 확인", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/timeseries_acf_pacf.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[ACF / PACF] 저장 완료 (lags = {lags})")

def plot_weekday_pattern(df):
    """
    요일별 감정지수 평균 및 분포 시각화.
    주간 패턴(월요병, 주말 효과 등)을 확인합니다.
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = ["월", "화", "수", "목", "금", "토", "일"]
    day_map = dict(zip(day_order, day_label))

    df_day = df[[TARGET_COL]].copy()
    df_day["요일"] = df_day.index.day_name().map(day_map)
    existing = [d for d in day_label if d in df_day["요일"].values]

    fig, axes = plt.subplots(1, 2, figsize = (13, 5))

    # 요일별 평균 막대그래프
    means = df_day.groupby("요일")[TARGET_COL].mean().reindex(existing)
    overall_mean = df_day[TARGET_COL].mean()
    bar_colors = ["crimson" if v >= overall_mean else "steelblue" for v in means.values]
    axes[0].bar(means.index, means.values, color = bar_colors, alpha = 0.8, edgecolor = "white")
    axes[0].axhline(y = overall_mean, color = "black", linestyle = "--", linewidth = 1.2, label = f"전체 평균 ({overall_mean:.1f})")
    axes[0].set_title("요일별 평균 감정지수\n(빨강: 평균 이상 / 파랑: 평균 미만)", fontsize = 11, weight = "bold")
    axes[0].set_ylabel(TARGET_COL)
    axes[0].legend(fontsize = 9)
    axes[0].set_ylim(0, 10)

    # 요일별 박스플롯
    sns.boxplot(data = df_day, x = "요일", y = TARGET_COL, order = existing, palette = "pastel", ax = axes[1])
    sns.stripplot(data = df_day, x = "요일", y = TARGET_COL, order = existing, color = "black", alpha = 0.4, jitter = 0.1, ax = axes[1])
    axes[1].axhline(y = overall_mean, color = "crimson", linestyle = "--", linewidth = 1.2, label = f"전체 평균 ({overall_mean:.1f})")
    axes[1].set_title("요일별 감정지수 분포", fontsize = 11, weight = "bold")
    axes[1].legend(fontsize = 9)

    plt.suptitle("주간 감정 패턴 분석", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/timeseries_weekday.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[요일별 패턴] 저장 완료")

def print_summary(y, df):
    """핵심 통계 요약 출력"""
    print("\n" + "=" * 50)
    print("[시계열 분석 핵심 통계 요약]")
    print("=" * 50)
    print(f"  분석 기간    : {y.index[0].date()} ~ {y.index[-1].date()} ({len(y)}일)")
    print(f"  감정지수 평균: {y.mean():.2f}")
    print(f"  감정지수 표준편차: {y.std():.2f}")
    print(f"  최고점: {y.max():.1f} ({y.idxmax().date()})")
    print(f"  최저점: {y.min():.1f} ({y.idxmin().date()})")

    # 주간 추세 (7일 이동평균 기울기)
    ma7 = y.rolling(window = 7, center = True).mean().dropna()
    if len(ma7) >= 2:
        x = np.arange(len(ma7))
        slope = np.polyfit(x, ma7.values, 1)[0]
        direction = "상승" if slope > 0.01 else "하락" if slope < -0.01 else "보합"
        print(f"  전반적 추세  : {direction} (7일 이동평균 기울기: {slope:.4f})")
    print("=" * 50)

if __name__ == "__main__":
    set_universal_font()

    parser = argparse.ArgumentParser(description = "시계열 분석 스크립트")
    parser.add_argument(
        "--input",
        type = str,
        default = "notion_brain_dump_raw_cleaned_impute_none.csv",
        help = "입력 CSV 파일 경로 (기본값: notion_brain_dump_raw_cleaned_impute_none.csv / 시계열은 none 스케일링 권장)"
    )
    parser.add_argument(
        "--period",
        type = int,
        default = 7,
        help = "시계열 분해 주기 (기본값: 7일 / 주간 패턴 기준)"
    )
    parser.add_argument(
        "--lags",
        type = int,
        default = 21,
        help = "ACF / PACF 최대 lag 수 (기본값: 21일)"
    )
    args = parser.parse_args()

    try:
        df, y = load_data(args.input)
        test_stationarity(y)
        plot_trend_with_ma(y)
        plot_decomposition(y, period = args.period)
        plot_acf_pacf(y, lags = args.lags)
        plot_weekday_pattern(df)
        print_summary(y, df)
        print(f"\n[최종 완료] 시계열 분석 시각화가 outputs/ 폴더에 저장되었습니다.")

    except FileNotFoundError:
        print(f"[오류] 파일이 없습니다: {args.input}")
        print("전처리 단계를 먼저 실행하거나 --input 경로를 확인해주세요.")
        print("시계열 분석은 none 스케일링 적용 파일 사용을 권장합니다.")
        print("예: python data_preprocessing.py --mode impute --outlier detect --scaler none")