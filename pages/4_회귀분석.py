import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_visualization import set_universal_font
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title = "회귀분석", page_icon = "📐", layout = "wide")

st.title("📐 회귀분석 (OLS + Ridge / Lasso)")
st.markdown("---")

EXCLUDE_COLS = ["감정지수", "오늘의 요약", "브레인 덤프"]

if st.session_state.get("cleaned_df") is None:
    st.warning("먼저 데이터 수집 및 전처리를 완료해주세요.")
    st.stop()

df = st.session_state["cleaned_df"].copy()
set_universal_font()

st.success(f"✅ 전처리된 데이터 로드 완료 — {len(df)}행 · 스케일링: {st.session_state.get('clean_suffix', '')}")
st.warning("💡 회귀분석은 standard 스케일링 데이터를 사용해야 VIF가 정확하게 계산됩니다.")

st.markdown("""
감정지수에 영향을 미치는 독립변수를 파악하고 예측 모델을 구축하는 분석입니다.
VIF > 10 변수가 감지되면 Ridge / Lasso 정규화 회귀도 자동 실행됩니다.
""")

if st.button("📐 회귀분석 실행", use_container_width = True, type = "primary"):

    # 원핫 인코딩
    cat_cols = [c for c in df.select_dtypes(exclude = [np.number]).columns if c not in EXCLUDE_COLS]
    if cat_cols:
        df = pd.get_dummies(df, columns = cat_cols, drop_first = True, dtype = int)

    feature_cols = [c for c in df.select_dtypes(include = [np.number]).columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df["감정지수"]

    st.markdown(f"**독립변수 {len(feature_cols)}개 / 관측치 {len(df)}행**")

    # VIF 계산
    st.markdown("---")
    st.markdown("## 📊 VIF (분산팽창지수)")
    vif_df = pd.DataFrame({
        "변수": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending = False).reset_index(drop = True)
    st.dataframe(vif_df, use_container_width = True)

    high_vif = vif_df[vif_df["VIF"] > 10]
    if not high_vif.empty:
        st.warning(f"⚠️ VIF > 10 변수 {len(high_vif)}개 감지 → Ridge / Lasso도 자동 실행됩니다.")
    else:
        st.success("✅ 모든 변수의 VIF < 10")

    # OLS
    st.markdown("---")
    st.markdown("## 📋 OLS 회귀분석 결과")
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    st.text(str(model.summary()))

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{model.rsquared:.3f}")
    col2.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
    col3.metric("F-통계량 p값", f"{model.f_pvalue:.4f}")

    sig_vars = model.pvalues[1:][model.pvalues[1:] < 0.05]
    if not sig_vars.empty:
        st.success(f"유의미한 변수 (p < 0.05): {sig_vars.index.tolist()}")
    else:
        st.info("유의미한 변수 없음 (데이터 부족 가능성)")

    st.markdown("---")

    # 회귀계수 시각화
    st.markdown("## 📊 회귀계수 (* p < 0.05)")
    coef_df = pd.DataFrame({
        "변수": feature_cols,
        "계수": model.params[1:].values,
        "p값": model.pvalues[1:].values
    }).sort_values("계수", key = abs, ascending = False)

    colors = ["crimson" if c > 0 else "steelblue" for c in coef_df["계수"]]
    labels = [f"{v} *" if p < 0.05 else v for v, p in zip(coef_df["변수"], coef_df["p값"])]

    fig, ax = plt.subplots(figsize = (8, max(5, len(feature_cols) * 0.45)))
    ax.barh(labels, coef_df["계수"], color = colors, alpha = 0.8, edgecolor = "white")
    ax.axvline(x = 0, color = "black", linewidth = 0.8, linestyle = "--")
    ax.set_title("회귀계수 (* p < 0.05)\n빨강: 감정지수 상승 / 파랑: 감정지수 하락", fontsize = 12, weight = "bold")
    ax.set_xlabel("회귀계수")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 실제값 vs 예측값
    st.markdown("## 🎯 실제값 vs 예측값")
    fitted = model.fittedvalues
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.scatter(y, fitted, alpha = 0.5, color = "steelblue", s = 40)
    min_val = min(y.min(), fitted.min()) - 0.5
    max_val = max(y.max(), fitted.max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], color = "crimson", linestyle = "--", linewidth = 1.5, label = "완벽한 예측선")
    ax.set_xlabel("실제 감정지수")
    ax.set_ylabel("예측 감정지수")
    ax.set_title(f"실제값 vs 예측값\nR² = {model.rsquared:.3f}", fontsize = 12, weight = "bold")
    ax.legend(fontsize = 9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 잔차 진단
    st.markdown("## 🔬 잔차 진단 플롯")
    residuals = model.resid
    std_resid = residuals / residuals.std()

    fig, axes = plt.subplots(2, 2, figsize = (12, 9))
    axes[0, 0].scatter(fitted, residuals, alpha = 0.5, color = "steelblue", s = 30)
    axes[0, 0].axhline(0, color = "crimson", linestyle = "--", linewidth = 1.2)
    axes[0, 0].set_title("잔차 vs 예측값", fontsize = 11)
    axes[0, 1].hist(residuals, bins = 15, color = "steelblue", alpha = 0.7, edgecolor = "white")
    axes[0, 1].set_title("잔차 분포", fontsize = 11)
    sm.qqplot(residuals, line = "s", ax = axes[1, 0], alpha = 0.5)
    axes[1, 0].set_title("Q-Q 플롯", fontsize = 11)
    axes[1, 1].scatter(fitted, std_resid, alpha = 0.5, color = "steelblue", s = 30)
    axes[1, 1].axhline(0, color = "crimson", linestyle = "--", linewidth = 1.2)
    axes[1, 1].axhline(2, color = "orange", linestyle = ":", linewidth = 1.2, label = "±2 기준선")
    axes[1, 1].axhline(-2, color = "orange", linestyle = ":", linewidth = 1.2)
    axes[1, 1].set_title("표준화 잔차", fontsize = 11)
    axes[1, 1].legend(fontsize = 9)
    plt.suptitle(f"잔차 진단 플롯  |  R² = {model.rsquared:.3f}  /  Adj. R² = {model.rsquared_adj:.3f}",
                 fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Ridge / Lasso (VIF > 10일 때)
    if not high_vif.empty:
        st.markdown("---")
        st.markdown("## ⚖️ OLS / Ridge / Lasso 회귀계수 비교")
        alphas = np.logspace(-3, 3, 100)
        ridge_cv = RidgeCV(alphas = alphas, cv = 5)
        ridge_cv.fit(X, y)
        lasso_cv = LassoCV(alphas = alphas, cv = 5, max_iter = 10000)
        lasso_cv.fit(X, y)

        ridge_r2 = r2_score(y, ridge_cv.predict(X))
        lasso_r2 = r2_score(y, lasso_cv.predict(X))

        col1, col2, col3 = st.columns(3)
        col1.metric("OLS R²", f"{model.rsquared:.3f}")
        col2.metric("Ridge R²", f"{ridge_r2:.3f}", help = f"최적 alpha: {ridge_cv.alpha_:.4f}")
        col3.metric("Lasso R²", f"{lasso_r2:.3f}", help = f"최적 alpha: {lasso_cv.alpha_:.4f}")

        zero_coef = X.columns[lasso_cv.coef_ == 0].tolist()
        if zero_coef:
            st.info(f"Lasso가 제거한 변수 ({len(zero_coef)}개): {zero_coef}")

        coef_compare = pd.DataFrame({
            "변수": feature_cols,
            "OLS": model.params[1:].values,
            "Ridge": ridge_cv.coef_,
            "Lasso": lasso_cv.coef_
        }).set_index("변수")
        coef_compare = coef_compare.reindex(coef_compare["OLS"].abs().sort_values(ascending = True).index)

        fig, axes = plt.subplots(1, 3, figsize = (15, max(5, len(feature_cols) * 0.45)), sharey = True)
        colors_map = {"OLS": "steelblue", "Ridge": "darkorange", "Lasso": "seagreen"}
        for ax, col in zip(axes, ["OLS", "Ridge", "Lasso"]):
            bar_colors = ["crimson" if v > 0 else colors_map[col] for v in coef_compare[col]]
            ax.barh(coef_compare.index, coef_compare[col], color = bar_colors, alpha = 0.8, edgecolor = "white")
            ax.axvline(x = 0, color = "black", linewidth = 0.8, linestyle = "--")
            ax.set_title(col, fontsize = 12, weight = "bold")
            ax.set_xlabel("회귀계수")
        plt.suptitle("OLS / Ridge / Lasso 회귀계수 비교", fontsize = 12, weight = "bold", y = 1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✅ 회귀분석 완료!")