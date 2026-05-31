import warnings
warnings.filterwarnings("ignore")

import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 시각화 이미지 저장 경로
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok = True)

# 분석 대상에서 제외할 컬럼
EXCLUDE_COLS = ["감정지수"]

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
    """전처리된 CSV 로드 후 PCA 대상 수치형 독립변수만 반환"""
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    # 수치형 컬럼만 추출 후 종속변수 제외
    numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    print(f"[데이터 로드] {len(df)}행 / PCA 대상 컬럼 {len(feature_cols)}개: {feature_cols}")
    return df, df[feature_cols]

def run_pca(feature_df):
    """StandardScaler 적용 후 전체 주성분 PCA 실행"""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)

    # 전체 주성분으로 실행 (스크리 플롯용)
    pca_full = PCA()
    pca_full.fit(scaled)

    return pca_full, scaled, scaler

def plot_scree(pca_full, feature_df):
    """
    스크리 플롯 + 누적 분산 설명량
    - 고유값(eigenvalue)이 1 이상인 주성분 수 기준선 표시 (Kaiser 기준)
    - 누적 분산 80% 기준선 표시
    """
    n = len(pca_full.explained_variance_ratio_)
    x = range(1, n + 1)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_) * 100
    eigenvalues = pca_full.explained_variance_

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    # 스크리 플롯 (고유값)
    axes[0].plot(x, eigenvalues, marker = "o", color = "steelblue", linewidth = 2)
    axes[0].axhline(y = 1, color = "crimson", linestyle = "--", linewidth = 1.5, label = "Kaiser 기준 (고유값 = 1)")
    axes[0].set_title("스크리 플롯 (고유값)", fontsize = 13, weight = "bold")
    axes[0].set_xlabel("주성분 번호")
    axes[0].set_ylabel("고유값 (Eigenvalue)")
    axes[0].legend(fontsize = 9)
    axes[0].set_xticks(list(x))
    axes[0].grid(True, linestyle = ":", alpha = 0.5)

    # 누적 분산 설명량
    axes[1].bar(x, pca_full.explained_variance_ratio_ * 100, color = "steelblue", alpha = 0.7, label = "개별 설명량")
    axes[1].plot(x, cumulative, marker = "o", color = "darkorange", linewidth = 2, label = "누적 설명량")
    axes[1].axhline(y = 80, color = "crimson", linestyle = "--", linewidth = 1.5, label = "80% 기준선")
    axes[1].set_title("주성분별 분산 설명량", fontsize = 13, weight = "bold")
    axes[1].set_xlabel("주성분 번호")
    axes[1].set_ylabel("분산 설명량 (%)")
    axes[1].set_xticks(list(x))
    axes[1].legend(fontsize = 9)
    axes[1].grid(True, linestyle = ":", alpha = 0.5)

    # Kaiser 기준 충족 주성분 수 출력
    n_kaiser = sum(eigenvalues >= 1)
    # 누적 분산 80% 충족 주성분 수
    n_80 = int(np.argmax(cumulative >= 80)) + 1

    plt.suptitle(
        f"PCA 주성분 선택 기준  |  Kaiser 기준: {n_kaiser}개  /  누적 분산 80%: {n_80}개",
        fontsize = 12, weight = "bold", y = 1.02
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/factor_scree.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[스크리 플롯] Kaiser 기준 주성분 수: {n_kaiser}개 / 누적 분산 80% 달성 주성분 수: {n_80}개")
    return n_kaiser

def plot_loading_heatmap(scaled, feature_df, n_components):
    """
    적재량(loading) 히트맵
    - 각 변수가 주성분에 얼마나 기여하는지 시각화
    - 절댓값 0.4 이상이면 해당 주성분과 연관성이 높은 변수로 해석
    """
    pca = PCA(n_components = n_components)
    pca.fit(scaled)

    # 적재량 = 고유벡터 * sqrt(고유값)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(
        loadings,
        index = feature_df.columns,
        columns = [f"PC{i + 1}\n({pca.explained_variance_ratio_[i] * 100:.1f}%)" for i in range(n_components)]
    )

    plt.figure(figsize = (max(6, n_components * 1.5), max(6, len(feature_df.columns) * 0.6)))
    sns.heatmap(
        loading_df,
        annot = True, fmt = ".2f", cmap = "coolwarm",
        center = 0, linewidths = 0.5,
        vmin = -1, vmax = 1,
        annot_kws = {"size": 9}
    )
    plt.title("PCA 적재량 히트맵\n(|값| ≥ 0.4: 해당 주성분과 강한 연관)", fontsize = 13, weight = "bold", pad = 15)
    plt.xlabel("주성분 (괄호 안: 분산 설명량)")
    plt.ylabel("변수")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/factor_loadings.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[적재량 히트맵] {n_components}개 주성분 기준 저장 완료")

    # 주성분별 주요 변수 출력
    print("\n--- 주성분별 주요 변수 (|적재량| ≥ 0.4) ---")
    for col in loading_df.columns:
        major = loading_df[loading_df[col].abs() >= 0.4][col].sort_values(key = abs, ascending = False)
        if not major.empty:
            print(f"\n{col}:")
            for var, val in major.items():
                direction = "↑" if val > 0 else "↓"
                print(f"  {direction} {var}: {val:.2f}")

    return loading_df

def plot_biplot(scaled, feature_df, n_components):
    """
    바이플롯 (PC1 vs PC2)
    - 점: 각 날짜의 위치 (주성분 공간에서)
    - 화살표: 각 변수의 방향과 강도
    - 같은 방향 화살표: 양의 상관관계 / 반대 방향: 음의 상관관계
    """
    if n_components < 2:
        print("[바이플롯] 주성분이 2개 미만이라 생략합니다.")
        return

    pca = PCA(n_components = n_components)
    scores = pca.fit_transform(scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize = (9, 7))

    # 관측치 산점도
    ax.scatter(scores[:, 0], scores[:, 1], alpha = 0.5, color = "steelblue", s = 30, label = "관측치")

    # 변수 화살표
    scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.7
    for i, var in enumerate(feature_df.columns):
        ax.annotate(
            "", xy = (loadings[i, 0] * scale, loadings[i, 1] * scale), xytext = (0, 0),
            arrowprops = dict(arrowstyle = "->", color = "crimson", lw = 1.5)
        )
        ax.text(
            loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
            var, fontsize = 9, color = "crimson", ha = "center"
        )

    ax.axhline(0, color = "gray", linestyle = "--", linewidth = 0.8, alpha = 0.5)
    ax.axvline(0, color = "gray", linestyle = "--", linewidth = 0.8, alpha = 0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize = 11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize = 11)
    ax.set_title("PCA 바이플롯 (PC1 vs PC2)\n(화살표 방향이 같을수록 양의 상관관계)", fontsize = 13, weight = "bold")
    ax.legend(fontsize = 9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/factor_biplot.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[바이플롯] 저장 완료")

if __name__ == "__main__":
    set_universal_font()

    import argparse
    parser = argparse.ArgumentParser(description = "PCA 기반 요인분석 스크립트")
    parser.add_argument(
        "--input",
        type = str,
        default = "notion_brain_dump_raw_cleaned_impute_none.csv",
        help = "입력 CSV 파일 경로 (기본값: notion_brain_dump_raw_cleaned_impute_none.csv)"
    )
    args = parser.parse_args()

    try:
        df, feature_df = load_data(args.input)
        pca_full, scaled, scaler = run_pca(feature_df)
        n_components = plot_scree(pca_full, feature_df)
        plot_loading_heatmap(scaled, feature_df, n_components)
        plot_biplot(scaled, feature_df, n_components)
        print(f"\n[최종 완료] 요인분석 시각화가 outputs/ 폴더에 저장되었습니다.")
    except FileNotFoundError:
        print(f"[오류] 파일이 없습니다: {args.input}. 전처리 단계를 먼저 실행하거나 --input 경로를 확인해주세요.")