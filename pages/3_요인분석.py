import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_visualization import set_universal_font
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title = "요인분석", page_icon = "🔍", layout = "wide")

st.title("🔍 요인분석 (PCA)")
st.markdown("---")

EXCLUDE_COLS = ["감정지수", "오늘의 요약", "브레인 덤프"]

if st.session_state.get("cleaned_df") is None:
    st.warning("먼저 데이터 수집 및 전처리를 완료해주세요.")
    st.stop()

df = st.session_state["cleaned_df"].copy()
set_universal_font()

st.success(f"✅ 전처리된 데이터 로드 완료 — {len(df)}행 · 스케일링: {st.session_state.get('clean_suffix', '')}")
st.info("💡 요인분석은 스케일링이 내부적으로 적용되므로 none 스케일링 파일을 사용해도 됩니다.")
st.markdown("""
여러 변수들 사이에 숨어있는 공통 패턴(잠재 요인)을 찾는 분석입니다.
변수 간 구조를 파악하여 이후 회귀분석의 변수 선택 근거로 활용합니다.
""")

if st.button("🔍 요인분석 실행", use_container_width = True, type = "primary"):
    feature_cols = [c for c in df.select_dtypes(include = [np.number]).columns if c not in EXCLUDE_COLS]

    if len(feature_cols) < 2:
        st.error("분석 가능한 수치형 독립변수가 2개 미만입니다.")
        st.stop()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    # 전체 PCA (스크리 플롯용)
    pca_full = PCA()
    pca_full.fit(scaled)

    eigenvalues = pca_full.explained_variance_
    cumulative = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_kaiser = sum(eigenvalues >= 1)
    n_80 = int(np.argmax(cumulative >= 80)) + 1
    n_components = n_kaiser

    st.markdown("---")

    # 스크리 플롯
    st.markdown("## 📉 스크리 플롯 (최적 주성분 수 탐색)")
    st.caption(f"Kaiser 기준: {n_kaiser}개 / 누적 분산 80% 기준: {n_80}개")

    n = len(pca_full.explained_variance_ratio_)
    x = range(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    axes[0].plot(list(x), eigenvalues, marker = "o", color = "steelblue", linewidth = 2)
    axes[0].axhline(y = 1, color = "crimson", linestyle = "--", linewidth = 1.5, label = "Kaiser 기준 (고유값 = 1)")
    axes[0].set_title("스크리 플롯 (고유값)", fontsize = 13, weight = "bold")
    axes[0].set_xlabel("주성분 번호")
    axes[0].set_ylabel("고유값 (Eigenvalue)")
    axes[0].legend(fontsize = 9)
    axes[0].set_xticks(list(x))
    axes[0].grid(True, linestyle = ":", alpha = 0.5)

    axes[1].bar(x, pca_full.explained_variance_ratio_ * 100, color = "steelblue", alpha = 0.7, label = "개별 설명량")
    axes[1].plot(x, cumulative, marker = "o", color = "darkorange", linewidth = 2, label = "누적 설명량")
    axes[1].axhline(y = 80, color = "crimson", linestyle = "--", linewidth = 1.5, label = "80% 기준선")
    axes[1].set_title("주성분별 분산 설명량", fontsize = 13, weight = "bold")
    axes[1].set_xlabel("주성분 번호")
    axes[1].set_ylabel("분산 설명량 (%)")
    axes[1].set_xticks(list(x))
    axes[1].legend(fontsize = 9)
    axes[1].grid(True, linestyle = ":", alpha = 0.5)

    plt.suptitle(
        f"PCA 주성분 선택 기준  |  Kaiser 기준: {n_kaiser}개  /  누적 분산 80%: {n_80}개",
        fontsize = 12, weight = "bold", y = 1.02
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 적재량 히트맵
    st.markdown("## 🔥 적재량 히트맵")
    st.caption("|값| ≥ 0.4이면 해당 주성분과 강한 연관이 있는 변수입니다.")

    pca = PCA(n_components = n_components)
    pca.fit(scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(
        loadings,
        index = feature_cols,
        columns = [f"PC{i + 1}\n({pca.explained_variance_ratio_[i] * 100:.1f}%)" for i in range(n_components)]
    )

    fig, ax = plt.subplots(figsize = (max(6, n_components * 1.5), max(6, len(feature_cols) * 0.6)))
    sns.heatmap(loading_df, annot = True, fmt = ".2f", cmap = "coolwarm",
                center = 0, linewidths = 0.5, vmin = -1, vmax = 1,
                annot_kws = {"size": 9}, ax = ax)
    ax.set_title("PCA 적재량 히트맵\n(|값| ≥ 0.4: 해당 주성분과 강한 연관)", fontsize = 13, weight = "bold", pad = 15)
    ax.set_xlabel("주성분 (괄호 안: 분산 설명량)")
    ax.set_ylabel("변수")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 주성분별 주요 변수 출력
    st.markdown("#### 📋 주성분별 주요 변수 (|적재량| ≥ 0.4)")
    for col in loading_df.columns:
        major = loading_df[loading_df[col].abs() >= 0.4][col].sort_values(key = abs, ascending = False)
        if not major.empty:
            st.markdown(f"**{col}**")
            for var, val in major.items():
                direction = "↑" if val > 0 else "↓"
                st.write(f"  {direction} {var}: {val:.2f}")

    st.markdown("---")

    # 바이플롯
    st.markdown("## 🗺️ 바이플롯 (PC1 vs PC2)")
    st.caption("화살표 방향이 같을수록 양의 상관관계, 반대일수록 음의 상관관계입니다.")

    if n_components >= 2:
        scores = pca.fit_transform(scaled)
        fig, ax = plt.subplots(figsize = (9, 7))
        ax.scatter(scores[:, 0], scores[:, 1], alpha = 0.5, color = "steelblue", s = 30, label = "관측치")
        scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.7
        for i, var in enumerate(feature_cols):
            ax.annotate("", xy = (loadings[i, 0] * scale, loadings[i, 1] * scale), xytext = (0, 0),
                        arrowprops = dict(arrowstyle = "->", color = "crimson", lw = 1.5))
            ax.text(loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
                    var, fontsize = 9, color = "crimson", ha = "center")
        ax.axhline(0, color = "gray", linestyle = "--", linewidth = 0.8, alpha = 0.5)
        ax.axvline(0, color = "gray", linestyle = "--", linewidth = 0.8, alpha = 0.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize = 11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize = 11)
        ax.set_title("PCA 바이플롯 (PC1 vs PC2)", fontsize = 13, weight = "bold")
        ax.legend(fontsize = 9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✅ 요인분석 완료!")