import warnings
warnings.filterwarnings("ignore")

import os
import platform
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 시각화 이미지 저장 경로
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok = True)

# 분석 제외 컬럼
EXCLUDE_COLS = ["감정지수", "오늘의 요약", "브레인 덤프"]

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

def is_mlb_derived(col_name):
    """
    MLB 인코딩 파생 컬럼 여부 판별.
    `범주_변수명_항목` 구조(언더스코어 2개 이상)인 컬럼만 MLB 파생으로 간주.
    체크박스에서 파생된 `범주_변수명` 구조(언더스코어 1개)는 제외 대상이 아님.

    판별 기준을 0/1 이진값이 아닌 컬럼명 구조로 사용하는 이유:
    - 체크박스 컬럼(예: 신체_활동여부)은 원래부터 0/1로 수집된 독립 변수로,
      클러스터링 거리 계산에서 의미 있는 정보를 담고 있음
    - MLB 파생 컬럼(예: 여가_취미유형_게임)은 하나의 다중선택 변수가 쪼개진 것으로,
      항목 간 음의 상관관계가 발생하여 거리 계산을 왜곡할 수 있음
    """
    return col_name.count("_") >= 2

def load_data(file_path, exclude_mlb = False):
    """
    전처리된 CSV 로드 및 클러스터링 대상 컬럼 추출.
    exclude_mlb=True 시 MLB 인코딩 파생 컬럼(언더스코어 2개 이상)을 제외하고 로드.
    """
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    feature_cols = [c for c in df.select_dtypes(include = [np.number]).columns if c not in EXCLUDE_COLS]

    if exclude_mlb:
        mlb_cols = [c for c in feature_cols if is_mlb_derived(c)]
        feature_cols = [c for c in feature_cols if c not in mlb_cols]
        if mlb_cols:
            print(f"[MLB 제외] {len(mlb_cols)}개 다중선택 파생 컬럼 제외: {mlb_cols}")

    # 범주 접두어 기준으로 정렬 (히트맵 해석 편의)
    feature_cols = sorted(feature_cols, key = lambda x: x.split("_")[0])

    X = df[feature_cols]
    y = df["감정지수"]

    print(f"[데이터 로드] {len(df)}행 / 클러스터링 대상 컬럼 {len(feature_cols)}개")
    return df, X, y, feature_cols

def find_optimal_k(X, k_range = range(2, 9)):
    """
    최적 클러스터 수(k) 탐색.
    - 엘보우 플롯: 관성(inertia)이 급격히 꺾이는 지점이 최적 k
    - 실루엣 점수: 1에 가까울수록 클러스터가 잘 분리됨 (0.5 이상이면 양호)
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    # 엘보우 플롯
    axes[0].plot(list(k_range), inertias, marker = "o", color = "steelblue", linewidth = 2)
    axes[0].set_title("엘보우 플롯\n(관성이 급격히 꺾이는 지점 = 최적 k)", fontsize = 12, weight = "bold")
    axes[0].set_xlabel("클러스터 수 (k)")
    axes[0].set_ylabel("관성 (Inertia)")
    axes[0].grid(True, linestyle = ":", alpha = 0.5)

    # 실루엣 점수
    best_k = list(k_range)[np.argmax(silhouettes)]
    axes[1].plot(list(k_range), silhouettes, marker = "o", color = "darkorange", linewidth = 2)
    axes[1].axhline(y = 0.5, color = "crimson", linestyle = "--", linewidth = 1.2, label = "0.5 기준선 (양호)")
    axes[1].axvline(x = best_k, color = "seagreen", linestyle = ":", linewidth = 1.5, label = f"최고점 k={best_k}")
    axes[1].set_title("실루엣 점수\n(높을수록 클러스터가 잘 분리됨)", fontsize = 12, weight = "bold")
    axes[1].set_xlabel("클러스터 수 (k)")
    axes[1].set_ylabel("실루엣 점수")
    axes[1].legend(fontsize = 9)
    axes[1].grid(True, linestyle = ":", alpha = 0.5)

    plt.suptitle(f"최적 클러스터 수 탐색  |  실루엣 점수 최고: k = {best_k}", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clustering_optimal_k.png", dpi = 300, bbox_inches = "tight")
    plt.close()

    print(f"\n[최적 k 탐색] 실루엣 점수 기준 최적 k = {best_k}")
    for k, s in zip(k_range, silhouettes):
        print(f"  k={k}: 실루엣 점수 {s:.3f}")

    return best_k

def run_kmeans(X, k):
    """K-Means 클러스터링 실행"""
    km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"\n[K-Means] k={k} / 실루엣 점수: {score:.3f}")
    return labels, km

def plot_cluster_heatmap(df, X, y, labels, feature_cols, k):
    """
    클러스터별 변수 평균 히트맵.
    범주 접두어 기준으로 변수를 정렬하여 범주별 패턴을 한눈에 파악.
    - 빨강: 평균보다 높음 / 파랑: 평균보다 낮음
    - 클러스터별 감정지수 평균도 함께 표시
    """
    df_c = X.copy()
    df_c["클러스터"] = labels
    df_c["감정지수"] = y.values

    # 클러스터별 변수 평균
    cluster_means = df_c.groupby("클러스터")[feature_cols].mean()

    # 전체 평균 대비 z-score로 색상 표현 (스케일 차이 제거)
    cluster_means_z = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-8)

    # 클러스터별 감정지수 평균 (제목에 표시)
    mood_means = df_c.groupby("클러스터")["감정지수"].mean()
    col_labels = [f"클러스터 {i}\n(감정지수 평균: {mood_means[i]:.1f})" for i in range(k)]

    fig, ax = plt.subplots(figsize = (max(8, k * 2.5), max(6, len(feature_cols) * 0.55)))
    sns.heatmap(cluster_means_z.T, annot = cluster_means.T.round(2), fmt = ".2f", cmap = "coolwarm", center = 0,
                linewidths = 0.5, ax = ax, annot_kws = {"size": 8}, xticklabels = col_labels)
    ax.set_title("클러스터별 변수 평균 히트맵\n(색상: 전체 평균 대비 / 숫자: 실제 평균값 / 범주별 정렬)", fontsize = 12, weight = "bold", pad = 15)
    ax.set_xlabel("")
    ax.set_ylabel("변수 (범주별 정렬)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clustering_heatmap.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[클러스터 히트맵] 저장 완료")

def plot_cluster_mood(df, y, labels, k):
    """
    클러스터별 감정지수 분포 박스플롯.
    각 클러스터가 어떤 감정 상태인지 직관적으로 확인.
    """
    df_c = pd.DataFrame({"감정지수": y.values, "클러스터": [f"클러스터 {l}" for l in labels]}, index = y.index)
    order = [f"클러스터 {i}" for i in range(k)]

    plt.figure(figsize = (max(6, k * 2), 5))
    sns.boxplot(data = df_c, x = "클러스터", y = "감정지수", order = order, palette = "Set2")
    sns.stripplot(data = df_c, x = "클러스터", y = "감정지수", order = order, color = "black", alpha = 0.4, jitter = 0.1)
    plt.axhline(y = y.mean(), color = "crimson", linestyle = "--", linewidth = 1.2, label = f"전체 평균 ({y.mean():.1f})")
    plt.title("클러스터별 감정지수 분포\n(박스 위치가 높을수록 긍정적인 하루 패턴)", fontsize = 12, weight = "bold")
    plt.legend(fontsize = 9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clustering_mood.png", dpi = 300)
    plt.close()
    print("[클러스터별 감정지수] 저장 완료")

def plot_cluster_pca(X, labels, k):
    """
    PCA 2D 시각화로 클러스터 분포 확인.
    점들이 잘 분리되어 있을수록 클러스터링이 잘 된 것.
    """
    pca = PCA(n_components = 2)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    plt.figure(figsize = (8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, k))

    for i in range(k):
        mask = labels == i
        plt.scatter(coords[mask, 0], coords[mask, 1], color = colors[i], label = f"클러스터 {i}", 
                    alpha = 0.7, s = 50, edgecolors = "white", linewidth = 0.5)

    plt.xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize = 11)
    plt.ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize = 11)
    plt.title("PCA 2D 클러스터 분포\n(점들이 잘 분리될수록 클러스터링 품질 높음)", fontsize = 12, weight = "bold")
    plt.legend(fontsize = 10)
    plt.grid(True, linestyle = ":", alpha = 0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clustering_pca.png", dpi = 300)
    plt.close()
    print("[PCA 클러스터 분포] 저장 완료")

def print_cluster_summary(df, X, y, labels, feature_cols, k):
    """클러스터별 핵심 특징 요약 출력"""
    df_c = X.copy()
    df_c["클러스터"] = labels
    df_c["감정지수"] = y.values
    df_c["날짜"] = y.index

    overall_mean = X.mean()

    print("\n" + "=" * 55)
    print("[클러스터별 특징 요약]")
    print("=" * 55)

    for i in range(k):
        mask = labels == i
        cluster_df = df_c[df_c["클러스터"] == i]
        mood_mean = cluster_df["감정지수"].mean()
        size = mask.sum()

        # 전체 평균 대비 가장 두드러진 변수 상위 3개
        diff = cluster_df[feature_cols].mean() - overall_mean
        top_high = diff.nlargest(3)
        top_low  = diff.nsmallest(3)

        print(f"\n클러스터 {i} ({size}일 / 감정지수 평균: {mood_mean:.1f})")
        print(f"  높은 변수: {', '.join([f'{v}(+{d:.2f})' for v, d in top_high.items()])}")
        print(f"  낮은 변수: {', '.join([f'{v}({d:.2f})' for v, d in top_low.items()])}")

    print("=" * 55)

if __name__ == "__main__":
    set_universal_font()

    parser = argparse.ArgumentParser(description = "K-Means 클러스터링 분석 스크립트")
    parser.add_argument(
        "--input",
        type = str,
        default = "notion_brain_dump_raw_cleaned_impute_minmax.csv",
        help = "입력 CSV 파일 경로 (기본값: notion_brain_dump_raw_cleaned_impute_minmax.csv / 클러스터링은 minmax 스케일링 권장)"
    )
    parser.add_argument(
        "--k",
        type = int,
        default = None,
        help = "클러스터 수 직접 지정 (기본값: 자동 탐색)"
    )
    parser.add_argument(
        "--exclude_mlb",
        action = "store_true",
        help = "MLB 인코딩된 0/1 이진 컬럼 제외 (다중선택 컬럼이 클러스터링을 지배할 때 사용)"
    )
    args = parser.parse_args()

    try:
        df, X, y, feature_cols = load_data(args.input, exclude_mlb = args.exclude_mlb)

        if args.k:
            k = args.k
            print(f"[클러스터 수] 사용자 지정: k = {k}")
        else:
            k = find_optimal_k(X)
            print(f"[클러스터 수] 자동 선택: k = {k}")

        labels, km = run_kmeans(X, k)
        plot_cluster_heatmap(df, X, y, labels, feature_cols, k)
        plot_cluster_mood(df, y, labels, k)
        plot_cluster_pca(X, labels, k)
        print_cluster_summary(df, X, y, labels, feature_cols, k)

        print(f"\n[최종 완료] 클러스터링 시각화가 outputs/ 폴더에 저장되었습니다.")

    except FileNotFoundError:
        print(f"[오류] 파일이 없습니다: {args.input}")
        print("전처리 단계를 먼저 실행하거나 --input 경로를 확인해주세요.")
        print("클러스터링은 minmax 스케일링 적용 파일 사용을 권장합니다.")
        print("예: python data_preprocessing.py --mode impute --outlier detect --scaler minmax")