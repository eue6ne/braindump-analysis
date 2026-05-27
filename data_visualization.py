import warnings
warnings.filterwarnings('ignore')

import platform
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 텍스트 마이닝 대상 컬럼 (시각화 대상에서 제외)
TEXT_COLS = ["브레인 덤프", "오늘의 요약"]

def set_universal_font():
    """운영체제(OS)를 자동으로 감지하여 한글 깨짐이 없도록 폰트를 설정하는 함수"""
    os_name = platform.system()
    if os_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif os_name == "Darwin": # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else: # Linux: 해당 폰트가 없을 경우 아래 설치 명령어 실행바랍니다.
          #        sudo apt install fonts-nanum
          #        rm -rf ~/.cache/matplotlib
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False # 마이너스 기호 깨짐 방지 설정

def is_multi_select(series):
    """다중 선택(Multi-select) 컬럼 여부 판별"""
    if series.dtype != "object":
        return False
    return series.str.contains(",", na = False).any()

def run_grouped_subplot_eda(file_path):
    print(f"[{file_path}] 기반 데이터 시각화 시작...")
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    target_col = "감정지수"
    if target_col not in df.columns:
        print(f"[오류] '{target_col}' 컬럼이 없습니다.")
        return

    # 수치형 컬럼 추출 (타겟 컬럼 제외)
    numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    # 범주형 컬럼 추출 (텍스트 컬럼, 다중선택 컬럼, 타겟 컬럼 제외)
    categorical_cols = [
        c for c in df.select_dtypes(exclude = [np.number]).columns
        if c != target_col
        and c not in TEXT_COLS
        and not is_multi_select(df[c])
    ]
    # 고유값 비율이 50% 이하인 컬럼만 박스플롯 대상으로 선정 (자유 텍스트성 컬럼 제외)
    viz_cat_cols = [c for c in categorical_cols if (df[c].nunique() / len(df[c].dropna())) <= 0.5]

    # 수치형 변수 간 상관관계 히트맵
    if len(numeric_cols) > 1:
        plt.figure(figsize = (12, 10))
        sns.heatmap(df[numeric_cols].corr(), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5, annot_kws = {"size": 8})
        plt.xticks(rotation = 45, ha = 'right', fontsize = 9) 
        plt.yticks(rotation = 0, fontsize = 9)
        plt.title("수치형 변수 간 상관관계 히트맵", fontsize = 14, weight = "bold", pad = 20)
        plt.tight_layout()
        plt.savefig("eda_heatmap.png", dpi = 300)
        plt.close()

    # 모든 수치형 변수의 분포 히스토그램
    num_count = len(numeric_cols)
    if num_count > 0:
        ncols = min(3, num_count)
        nrows = math.ceil(num_count / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
        axes = axes.flatten() if num_count > 1 else [axes]

        i = -1 # 컬럼이 0개일 때 아래 range(i + 1, ...) 오류 방지
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde = True, ax = axes[i], color = "teal", bins = 10)
            axes[i].set_title(f"[{col}] 분포", fontsize = 11)
            axes[i].set_ylabel("")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("수치형 변수별 분포 히스토그램", fontsize = 14, weight = "bold", y = 1.02)
        plt.tight_layout()
        plt.savefig("eda_histograms.png", dpi = 300, bbox_inches = "tight")
        plt.close()

    # 모든 범주형 변수별 감정지수 분포 박스플롯
    cat_count = len(viz_cat_cols)
    if cat_count > 0:
        ncols = min(3, cat_count)
        nrows = math.ceil(cat_count / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
        axes = axes.flatten() if cat_count > 1 else [axes]

        i = -1 # 컬럼이 0개일 때 아래 range(i + 1, ...) 오류 방지
        for i, col in enumerate(viz_cat_cols):
            sns.boxplot(data = df, x = col, y = target_col, ax = axes[i], palette = "Set3")
            sns.stripplot(data = df, x = col, y = target_col, ax = axes[i], color = "black", alpha = 0.3, jitter = 0.1)
            axes[i].set_title(f"[{col}]별 {target_col} 분포", fontsize = 11)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("범주형 변수별 감정지수 비교 박스플롯", fontsize = 14, weight = "bold", y = 1.02)
        plt.tight_layout()
        plt.savefig("eda_boxplots.png", dpi = 300, bbox_inches = "tight")
        plt.close()

    # 종속변수 시계열 흐름 및 회귀 추세선
    plt.figure(figsize = (10, 5))
    plt.plot(df.index, df[target_col], marker = "o", color = "lightgray", markersize = 5, linewidth = 1.2, label = f"실제 {target_col}")
    x_ord = np.array([date.toordinal() for date in df.index])
    slope, intercept = np.polyfit(x_ord, df[target_col].values, 1)
    trend_color = "crimson" if slope >= 0 else "dodgerblue"
    plt.plot(df.index, slope * x_ord + intercept, color = trend_color, linewidth = 2.5, label = f"추세선 (기울기: {slope:.4f})")
    plt.title(f"시간 흐름에 따른 {target_col} 추세선 분석", fontsize = 12, pad = 10)
    plt.grid(True, linestyle = ":", alpha = 0.5)
    plt.legend(fontsize = 10)
    plt.xticks(rotation = 35)
    plt.tight_layout()
    plt.savefig("eda_target_trend.png", dpi = 300)
    plt.close()

    # 다중선택 컬럼 항목별 감정지수 평균 막대그래프
    # MLB 인코딩으로 분리된 컬럼 탐지: 수치형 컬럼 중 원본 컬럼명_항목 형태인 것
    # 원본 다중선택 컬럼명 추출 (언더스코어 기준 앞부분이 동일한 그룹)
    encoded_groups = {}
    for col in df.select_dtypes(include = [np.number]).columns:
        if col == target_col or "_" not in col:
            continue
        # 범주_변수명_항목 구조에서 마지막 언더스코어 이전을 원본 컬럼명으로 간주
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            group, item = parts
            # 그룹 내 고유값이 2개 이상이고 0/1로만 구성된 경우 MLB 인코딩 컬럼으로 판단
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                if group not in encoded_groups:
                    encoded_groups[group] = []
                encoded_groups[group].append((item, col))

    for group, items in encoded_groups.items():
        if len(items) < 2:
            continue
        item_labels = [item for item, _ in items]
        item_means  = [df[col].mul(df[target_col]).sum() / df[col].sum()
                       if df[col].sum() > 0 else 0
                       for _, col in items]

        fig, ax = plt.subplots(figsize = (max(6, len(items) * 1.2), 4))
        bars = ax.bar(item_labels, item_means, color = "steelblue", alpha = 0.8, edgecolor = "white")
        ax.axhline(y = df[target_col].mean(), color = "crimson", linestyle = "--",
                   linewidth = 1.5, label = f"전체 평균 ({df[target_col].mean():.2f})")
        ax.set_title(f"[{group}] 항목별 평균 {target_col}", fontsize = 12, weight = "bold")
        ax.set_ylabel(target_col)
        ax.set_ylim(0, 10)
        ax.legend(fontsize = 9)
        plt.xticks(rotation = 30, ha = "right")
        plt.tight_layout()
        safe_name = group.replace("/", "_").replace(" ", "_")
        plt.savefig(f"eda_multiselect_{safe_name}.png", dpi = 300, bbox_inches = "tight")
        plt.close()

    # 수치형 변수 페어플롯 (수치형 컬럼 수가 2개 이상일 때)
    pair_cols = [c for c in numeric_cols if c not in [col for _, col in
                 [pair for group_items in encoded_groups.values() for pair in group_items]]]
    pair_cols = [target_col] + pair_cols  # 종속변수 포함

    if len(pair_cols) >= 3:
        pair_df = df[pair_cols].copy()
        pair_plot = sns.pairplot(pair_df, diag_kind = "kde", plot_kws = {"alpha": 0.5, "color": "teal"},
                                 diag_kws = {"color": "teal"})
        pair_plot.figure.suptitle("수치형 변수 페어플롯", fontsize = 14, weight = "bold", y = 1.02)
        pair_plot.savefig("eda_pairplot.png", dpi = 300, bbox_inches = "tight")
        plt.close()

    # 요일별 감정지수 박스플롯
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = ["월", "화", "수", "목", "금", "토", "일"]
    day_map   = dict(zip(day_order, day_label))

    df_day = df.copy()
    df_day["요일"] = df_day.index.day_name().map(day_map)
    existing_days = [d for d in day_label if d in df_day["요일"].values]

    if len(existing_days) >= 2:
        plt.figure(figsize = (9, 4))
        sns.boxplot(data = df_day, x = "요일", y = target_col,
                    order = existing_days, palette = "pastel")
        sns.stripplot(data = df_day, x = "요일", y = target_col,
                      order = existing_days, color = "black", alpha = 0.4, jitter = 0.1)
        plt.title(f"요일별 {target_col} 분포", fontsize = 12, weight = "bold")
        plt.tight_layout()
        plt.savefig("eda_weekday.png", dpi = 300)
        plt.close()

    print("- [성공] 테마별 시각화 이미지 저장 완료")



if __name__ == "__main__":
    set_universal_font()

    parser = argparse.ArgumentParser(description = "노션 브레인 덤프 데이터 EDA 시각화 스크립트")
    parser.add_argument(
        "--data",
        type = str,
        choices = ["drop", "impute"],
        default = "impute",
        help = "분석에 사용할 전처리 데이터 타입 선택 (기본값: impute)"
    )
    parser.add_argument(
        "--scaler",
        type = str,
        choices = ["standard", "minmax", "none"],
        default = "none",
        help = "분석에 사용할 스케일링 타입 선택 (기본값: none)"
    )
    args = parser.parse_args()

    target_file = f"notion_brain_dump_cleaned_{args.data}_{args.scaler}.csv"
    try:
        run_grouped_subplot_eda(target_file)
        print(f"\n[최종 완료] {target_file} 기반의 시각화가 성공적으로 완료되었습니다!")
    except FileNotFoundError:
        print(f"[오류] 파일이 없습니다: {target_file}. 전처리 단계를 먼저 실행하세요.")