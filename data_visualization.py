import warnings
warnings.filterwarnings('ignore')

import platform
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_universal_font():
    """운영체제(OS)를 자동으로 감지하여 한글 깨짐이 없도록 폰트를 설정하는 함수"""
    os_name = platform.system()
    if os_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif os_name == "Darwin": # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else: # Linux
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False # 마이너스 기호 깨짐 방지 설정

def run_grouped_subplot_eda(file_path):
    print(f"[{file_path}] 기반 데이터 시각화 시작...")
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    target_col = "감정지수"
    if target_col not in df.columns:
        print(f"[오류] '{target_col}' 컬럼이 없습니다."); return
    
    # 수치형 컬럼 추출 (타입이 숫자이며 타겟 컬럼이 아닌 것)
    numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    # 범주형 컬럼 추출 (다중 선택 제외)
    categorical_cols = [c for c in df.select_dtypes(exclude = [np.number]).columns 
                        if c != target_col]
    viz_cat_cols = [c for c in categorical_cols if (df[c].nunique() / len(df[c].dropna())) <= 0.5]

    # 수치형 변수 간 상관관계 히트맵
    if len(numeric_cols) > 1:
        plt.figure(figsize = (8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
        plt.title("수치형 변수 간 상관관계 히트맵", fontsize = 14, weight = "bold")
        plt.tight_layout(); plt.savefig("eda_heatmap.png", dpi = 300); plt.close()

    # 모든 수치형 변수의 분포 히스토그램
    num_count = len(numeric_cols)
    if num_count > 0:
        ncols = min(3, num_count); nrows = math.ceil(num_count / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
        axes = axes.flatten() if num_count > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde = True, ax = axes[i], color = "teal", bins = 10)
            axes[i].set_title(f"[{col}] 분포", fontsize = 11); axes[i].set_ylabel("")
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        
        plt.suptitle("수치형 변수별 분포 히스토그램", fontsize = 14, weight = "bold", y = 1.02)
        plt.tight_layout()
        plt.savefig("eda_histograms.png", dpi = 300, bbox_inches = "tight")
        plt.close()

    # 모든 범주형 변수별 감정지수 분포 박스플롯
    cat_count = len(viz_cat_cols)
    if cat_count > 0:
        ncols = min(3, cat_count); nrows = math.ceil(cat_count / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
        axes = axes.flatten() if cat_count > 1 else [axes]
        
        for i, col in enumerate(viz_cat_cols):
            sns.boxplot(data = df, x = col, y = target_col, ax = axes[i], palette = "Set3")
            sns.stripplot(data = df, x = col, y = target_col, ax = axes[i], color = "black", alpha = 0.3, jitter = 0.1)
            axes[i].set_title(f"[{col}]별 {target_col} 분포", fontsize = 11)
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        
        plt.suptitle("범주형 변수별 감정지수 비교 박스플롯", fontsize = 14, weight = "bold", y = 1.02)
        plt.tight_layout(); plt.savefig("eda_boxplots.png", dpi = 300, bbox_inches = "tight"); plt.close()

    # 종속변수 시계열 흐름 및 회귀 추세선 (단일 저장)
    plt.figure(figsize = (10, 5))
    plt.plot(df.index, df[target_col], marker = "o", color = "lightgray", markersize = 5, linewidth = 1.2, label = "실제 기분")
    x_ord = np.array([date.toordinal() for date in df.index])
    slope, intercept = np.polyfit(x_ord, df[target_col].values, 1)
    plt.plot(df.index, slope * x_ord + intercept, color = "crimson" if slope >= 0 else "dodgerblue", linewidth = 2.5, label = f"추세선 (기울기: {slope:.4f})")
    plt.title(f"시간 흐름에 따른 {target_col} 추세선 분석", fontsize = 12, pad = 10)
    plt.grid(True, linestyle = ":", alpha = 0.5); plt.legend(fontsize = 10); plt.xticks(rotation = 35)
    plt.tight_layout(); plt.savefig("eda_target_trend.png", dpi = 300); plt.close()
    print("- [성공] 테마별 시각화 이미지 3종 저장 완료")

if __name__ == "__main__":
    set_universal_font()
    
    parser = argparse.ArgumentParser(
        description = "노션 브레인 덤프 데이터 EDA 시각화 스크립트"
    )
    parser.add_argument(
        "--data", 
        type = str, 
        choices = ["drop", "impute"], 
        default = "impute", 
        help = "분석에 사용할 전처리 데이터 타입 선택 (기본값: impute)"
    )
    args = parser.parse_args()
    
    # 파일 매핑 및 실행부
    target_file = f"notion_brain_dump_cleaned_{args.data}.csv"
    try:
        run_grouped_subplot_eda(target_file)
        print(f"\n[최종 완료] {target_file} 기반의 시각화가 성공적으로 완료되었습니다!")
    except FileNotFoundError:
        print(f"[오류] 파일이 없습니다: {target_file}. 전처리 단계를 먼저 실행하세요.")