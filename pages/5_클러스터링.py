import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_visualization import set_universal_font
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title = "클러스터링", page_icon = "🗂️", layout = "wide")

st.title("🗂️ 클러스터링 (K-Means)")
st.markdown("---")

EXCLUDE_COLS = ["감정지수", "오늘의 요약", "브레인 덤프"]

if st.session_state.get("cleaned_df") is None:
    st.warning("먼저 데이터 수집 및 전처리를 완료해주세요.")
    st.stop()

df = st.session_state["cleaned_df"].copy()
set_universal_font()

st.success(f"✅ 전처리된 데이터 로드 완료 — {len(df)}행 · 스케일링: {st.session_state.get('clean_suffix', '')}")
st.info("💡 클러스터링은 minmax 스케일링 데이터를 사용하는 것을 권장합니다.")

st.markdown("""
하루 패턴을 유형별로 그룹화하는 분석입니다.
비슷한 신체 컨디션·업무 패턴·여가 활동을 가진 날끼리 묶어 감정 유형을 분류합니다.
""")

# 옵션
col1, col2 = st.columns(2)
with col1:
    exclude_mlb = st.checkbox(
        "다중선택 파생 컬럼 제외 (`--exclude_mlb`)",
        value = True,
        help = "언더스코어 2개 이상인 MLB 인코딩 컬럼을 제외합니다. 클러스터가 취미 종류로만 분리될 때 사용하세요."
    )
with col2:
    k_manual = st.number_input("클러스터 수 직접 지정 (0 = 자동 탐색)", min_value = 0, max_value = 20, value = 0)

if st.button("🗂️ 클러스터링 실행", use_container_width = True, type = "primary"):

    feature_cols = [c for c in df.select_dtypes(include = [np.number]).columns if c not in EXCLUDE_COLS]

    if exclude_mlb:
        mlb_cols = [c for c in feature_cols if c.count("_") >= 2]
        feature_cols = [c for c in feature_cols if c not in mlb_cols]
        if mlb_cols:
            st.info(f"MLB 제외: {mlb_cols}")

    feature_cols = sorted(feature_cols, key = lambda x: x.split("_")[0])
    X = df[feature_cols]
    y = df["감정지수"]

    st.markdown(f"**클러스터링 대상 컬럼 {len(feature_cols)}개**")

    # 최적 k 탐색
    st.markdown("---")
    st.markdown("## 📉 최적 클러스터 수 탐색")

    k_range = range(2, 9)
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = list(k_range)[np.argmax(silhouettes)]
    k = k_manual if k_manual > 0 else best_k
    st.caption(f"실루엣 점수 최고: k = {best_k} / 사용할 k = {k}")

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    axes[0].plot(list(k_range), inertias, marker = "o", color = "steelblue", linewidth = 2)
    axes[0].set_title("엘보우 플롯", fontsize = 12, weight = "bold")
    axes[0].set_xlabel("클러스터 수 (k)")
    axes[0].set_ylabel("관성 (Inertia)")
    axes[0].grid(True, linestyle = ":", alpha = 0.5)

    axes[1].plot(list(k_range), silhouettes, marker = "o", color = "darkorange", linewidth = 2)
    axes[1].axhline(y = 0.5, color = "crimson", linestyle = "--", linewidth = 1.2, label = "0.5 기준선")
    axes[1].axvline(x = best_k, color = "seagreen", linestyle = ":", linewidth = 1.5, label = f"최고점 k={best_k}")
    axes[1].set_title("실루엣 점수", fontsize = 12, weight = "bold")
    axes[1].set_xlabel("클러스터 수 (k)")
    axes[1].set_ylabel("실루엣 점수")
    axes[1].legend(fontsize = 9)
    axes[1].grid(True, linestyle = ":", alpha = 0.5)
    plt.suptitle(f"최적 클러스터 수 탐색  |  실루엣 점수 최고: k = {best_k}", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # K-Means 실행
    km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    st.info(f"k = {k} / 실루엣 점수: {score:.3f}")

    df_c = X.copy()
    df_c["클러스터"] = labels
    df_c["감정지수"] = y.values

    st.markdown("---")

    # 히트맵
    st.markdown("## 🔥 클러스터별 변수 평균 히트맵")
    cluster_means = df_c.groupby("클러스터")[feature_cols].mean()
    cluster_means_z = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-8)
    mood_means = df_c.groupby("클러스터")["감정지수"].mean()
    col_labels = [f"클러스터 {i}\n(감정지수 평균: {mood_means[i]:.1f})" for i in range(k)]

    fig, ax = plt.subplots(figsize = (max(8, k * 2.5), max(6, len(feature_cols) * 0.55)))
    sns.heatmap(cluster_means_z.T, annot = cluster_means.T.round(2), fmt = ".2f",
                cmap = "coolwarm", center = 0, linewidths = 0.5, ax = ax,
                annot_kws = {"size": 8}, xticklabels = col_labels)
    ax.set_title("클러스터별 변수 평균 히트맵\n(색상: 전체 평균 대비 / 숫자: 실제 평균값)", fontsize = 12, weight = "bold", pad = 15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 감정지수 박스플롯
    st.markdown("## 😊 클러스터별 감정지수 분포")
    df_mood = pd.DataFrame({"감정지수": y.values, "클러스터": [f"클러스터 {l}" for l in labels]}, index = y.index)
    order = [f"클러스터 {i}" for i in range(k)]
    fig, ax = plt.subplots(figsize = (max(6, k * 2), 5))
    sns.boxplot(data = df_mood, x = "클러스터", y = "감정지수", order = order, palette = "Set2", ax = ax)
    sns.stripplot(data = df_mood, x = "클러스터", y = "감정지수", order = order,
                  color = "black", alpha = 0.4, jitter = 0.1, ax = ax)
    ax.axhline(y = y.mean(), color = "crimson", linestyle = "--", linewidth = 1.2, label = f"전체 평균 ({y.mean():.1f})")
    ax.set_title("클러스터별 감정지수 분포", fontsize = 12, weight = "bold")
    ax.legend(fontsize = 9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # PCA 2D
    st.markdown("## 🗺️ PCA 2D 클러스터 분포")
    pca = PCA(n_components = 2)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize = (8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, k))
    for i in range(k):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], color = colors[i],
                   label = f"클러스터 {i}", alpha = 0.7, s = 50, edgecolors = "white", linewidth = 0.5)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize = 11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize = 11)
    ax.set_title("PCA 2D 클러스터 분포", fontsize = 12, weight = "bold")
    ax.legend(fontsize = 10)
    ax.grid(True, linestyle = ":", alpha = 0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 클러스터별 요약
    st.markdown("---")
    st.markdown("## 📋 클러스터별 특징 요약")
    overall_mean = X.mean()
    for i in range(k):
        cluster_df = df_c[df_c["클러스터"] == i]
        mood_mean = cluster_df["감정지수"].mean()
        size = (labels == i).sum()
        diff = cluster_df[feature_cols].mean() - overall_mean
        top_high = diff.nlargest(3)
        top_low = diff.nsmallest(3)
        with st.expander(f"클러스터 {i} — {size}일 / 감정지수 평균: {mood_mean:.1f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**높은 변수**")
                for var, d in top_high.items():
                    st.write(f"↑ {var}: +{d:.2f}")
            with col2:
                st.markdown("**낮은 변수**")
                for var, d in top_low.items():
                    st.write(f"↓ {var}: {d:.2f}")

    st.success("✅ 클러스터링 완료!")