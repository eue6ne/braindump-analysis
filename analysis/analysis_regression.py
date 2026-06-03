import warnings
warnings.filterwarnings("ignore")

import os
import platform
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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

def load_data(file_path):
    """전처리된 CSV 로드 및 범주형 컬럼 원핫 인코딩"""
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    # 범주형 컬럼 원핫 인코딩 (텍스트 컬럼 제외)
    cat_cols = [c for c in df.select_dtypes(exclude = [np.number]).columns if c not in EXCLUDE_COLS]

    if cat_cols:
        df = pd.get_dummies(df, columns = cat_cols, drop_first = True, dtype = int)
        print(f"[원핫 인코딩] {cat_cols} → {len(df.columns)}개 컬럼")

    # 독립변수 / 종속변수 분리
    feature_cols = [c for c in df.select_dtypes(include = [np.number]).columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df["감정지수"]

    print(f"[데이터 로드] {len(df)}행 / 독립변수 {len(feature_cols)}개")
    return df, X, y, feature_cols

def check_vif(X):
    """
    분산팽창지수(VIF) 계산으로 다중공선성 확인.
    VIF > 10: 다중공선성 문제 가능성 높음 → 해당 변수 제거 또는 PCA 고려
    VIF 5~10: 주의 필요
    VIF < 5: 양호
    """
    vif_df = pd.DataFrame()
    vif_df["변수"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df = vif_df.sort_values("VIF", ascending = False).reset_index(drop = True)

    print("\n--- VIF (분산팽창지수) ---")
    print(vif_df.to_string(index = False))

    high_vif = vif_df[vif_df["VIF"] > 10]
    if not high_vif.empty:
        print(f"\n[경고] VIF > 10 변수 ({len(high_vif)}개): {high_vif['변수'].tolist()}")
        print("  → 해당 변수의 다중공선성을 확인하세요.")
    else:
        print("\n[양호] 모든 변수의 VIF < 10")

    return vif_df

def run_ols(X, y):
    """
    OLS(최소자승법) 회귀분석 실행.
    statsmodels의 summary로 p-value, 계수, R² 등 상세 결과 출력.
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print("\n--- OLS 회귀분석 결과 ---")
    print(model.summary())
    return model

def plot_coefficients(model, feature_cols):
    """
    회귀계수 시각화.
    - 양수(빨강): 해당 변수 증가 시 감정지수 상승
    - 음수(파랑): 해당 변수 증가 시 감정지수 하락
    - * 표시: p-value < 0.05 (통계적으로 유의미)
    """
    coef_df = pd.DataFrame({
        "변수": feature_cols,
        "계수": model.params[1:].values,
        "p값": model.pvalues[1:].values
    }).sort_values("계수", key = abs, ascending = False)

    colors = ["crimson" if c > 0 else "steelblue" for c in coef_df["계수"]]
    labels = [f"{v} *" if p < 0.05 else v for v, p in zip(coef_df["변수"], coef_df["p값"])]

    plt.figure(figsize = (8, max(5, len(feature_cols) * 0.45)))
    bars = plt.barh(labels, coef_df["계수"], color = colors, alpha = 0.8, edgecolor = "white")
    plt.axvline(x = 0, color = "black", linewidth = 0.8, linestyle = "--")
    plt.title("회귀계수 (* p < 0.05)\n빨강: 감정지수 상승 / 파랑: 감정지수 하락", fontsize = 12, weight = "bold")
    plt.xlabel("회귀계수")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regression_coefficients.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[회귀계수 시각화] 저장 완료")

def plot_residuals(model, X, y):
    """
    잔차 진단 4종 플롯.
    - 잔차 vs 예측값: 패턴 없이 무작위로 퍼져있어야 함 (이분산성 확인)
    - 잔차 분포: 정규분포에 가까울수록 모델 가정 충족
    - Q-Q 플롯: 점들이 대각선에 가까울수록 정규성 충족
    - 표준화 잔차: |값| > 2 이상이면 이상치 가능성
    """
    X_const = sm.add_constant(X)
    fitted = model.fittedvalues
    residuals = model.resid
    std_resid = residuals / residuals.std()

    fig, axes = plt.subplots(2, 2, figsize = (12, 9))

    # 잔차 vs 예측값
    axes[0, 0].scatter(fitted, residuals, alpha = 0.5, color = "steelblue", s = 30)
    axes[0, 0].axhline(0, color = "crimson", linestyle = "--", linewidth = 1.2)
    axes[0, 0].set_xlabel("예측값")
    axes[0, 0].set_ylabel("잔차")
    axes[0, 0].set_title("잔차 vs 예측값\n(무작위 분포가 이상적)", fontsize = 11)

    # 잔차 분포
    axes[0, 1].hist(residuals, bins = 15, color = "steelblue", alpha = 0.7, edgecolor = "white")
    axes[0, 1].set_xlabel("잔차")
    axes[0, 1].set_ylabel("빈도")
    axes[0, 1].set_title("잔차 분포\n(정규분포에 가까울수록 양호)", fontsize = 11)

    # Q-Q 플롯
    sm.qqplot(residuals, line = "s", ax = axes[1, 0], alpha = 0.5)
    axes[1, 0].set_title("Q-Q 플롯\n(대각선에 가까울수록 정규성 충족)", fontsize = 11)

    # 표준화 잔차
    axes[1, 1].scatter(fitted, std_resid, alpha = 0.5, color = "steelblue", s = 30)
    axes[1, 1].axhline(0, color = "crimson", linestyle = "--", linewidth = 1.2)
    axes[1, 1].axhline(2, color = "orange", linestyle = ":", linewidth = 1.2, label = "±2 기준선")
    axes[1, 1].axhline(-2, color = "orange", linestyle = ":", linewidth = 1.2)
    axes[1, 1].set_xlabel("예측값")
    axes[1, 1].set_ylabel("표준화 잔차")
    axes[1, 1].set_title("표준화 잔차\n(|값| > 2: 이상치 가능성)", fontsize = 11)
    axes[1, 1].legend(fontsize = 9)

    plt.suptitle(f"잔차 진단 플롯  |  R² = {model.rsquared:.3f}  /  Adj. R² = {model.rsquared_adj:.3f}", 
                 fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regression_residuals.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[잔차 진단] 저장 완료")

def plot_actual_vs_predicted(model, y):
    """
    실제값 vs 예측값 산점도.
    점들이 대각선에 가까울수록 모델 예측력이 높음.
    """
    fitted = model.fittedvalues

    plt.figure(figsize = (6, 6))
    plt.scatter(y, fitted, alpha = 0.5, color = "steelblue", s = 40)
    min_val = min(y.min(), fitted.min()) - 0.5
    max_val = max(y.max(), fitted.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color = "crimson", linestyle = "--", linewidth = 1.5, label = "완벽한 예측선")
    plt.xlabel("실제 감정지수")
    plt.ylabel("예측 감정지수")
    plt.title(f"실제값 vs 예측값\n(대각선에 가까울수록 예측력 높음)\nR² = {model.rsquared:.3f}", fontsize = 12, weight = "bold")
    plt.legend(fontsize = 9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regression_actual_vs_predicted.png", dpi = 300)
    plt.close()
    print("[실제값 vs 예측값] 저장 완료")

def run_ridge_lasso(X, y):
    """
    Ridge / Lasso 정규화 회귀분석 실행.
    - Ridge : 모든 변수를 유지하면서 계수를 축소 (다중공선성에 강함)
    - Lasso : 불필요한 변수의 계수를 0으로 만들어 자동 변수 선택
    최적 alpha(정규화 강도)는 Cross-Validation으로 자동 탐색.
    """
    alphas = np.logspace(-3, 3, 100)

    # Ridge CV
    ridge_cv = RidgeCV(alphas = alphas, cv = 5)
    ridge_cv.fit(X, y)
    ridge_pred = ridge_cv.predict(X)
    ridge_r2 = r2_score(y, ridge_pred)

    # Lasso CV
    lasso_cv = LassoCV(alphas = alphas, cv = 5, max_iter = 10000)
    lasso_cv.fit(X, y)
    lasso_pred = lasso_cv.predict(X)
    lasso_r2 = r2_score(y, lasso_pred)

    # Lasso가 0으로 만든 변수
    zero_coef = X.columns[lasso_cv.coef_ == 0].tolist()

    print(f"\n--- Ridge 회귀 결과 ---")
    print(f"  최적 alpha: {ridge_cv.alpha_:.4f}")
    print(f"  R²: {ridge_r2:.3f}")

    print(f"\n--- Lasso 회귀 결과 ---")
    print(f"  최적 alpha: {lasso_cv.alpha_:.4f}")
    print(f"  R²: {lasso_r2:.3f}")
    if zero_coef:
        print(f"  제거된 변수 ({len(zero_coef)}개): {zero_coef}")
    else:
        print("  제거된 변수 없음")

    return ridge_cv, lasso_cv, ridge_r2, lasso_r2


def plot_regularized_coefficients(ols_model, ridge_model, lasso_model, feature_cols):
    """
    OLS / Ridge / Lasso 회귀계수 비교 시각화.
    - 세 모델의 계수를 나란히 비교해 다중공선성 영향을 확인
    - Lasso에서 0이 된 변수는 모델이 불필요하다고 판단한 것
    """
    coef_df = pd.DataFrame({
        "변수": feature_cols,
        "OLS":   ols_model.params[1:].values,
        "Ridge": ridge_model.coef_,
        "Lasso": lasso_model.coef_
    }).set_index("변수")

    # 절댓값 기준 내림차순 정렬 (OLS 기준)
    coef_df = coef_df.reindex(coef_df["OLS"].abs().sort_values(ascending = True).index)

    fig, axes = plt.subplots(1, 3, figsize = (15, max(5, len(feature_cols) * 0.45)), sharey = True)
    colors_map = {"OLS": "steelblue", "Ridge": "darkorange", "Lasso": "seagreen"}

    for ax, col in zip(axes, ["OLS", "Ridge", "Lasso"]):
        bar_colors = ["crimson" if v > 0 else colors_map[col] for v in coef_df[col]]
        ax.barh(coef_df.index, coef_df[col], color = bar_colors, alpha = 0.8, edgecolor = "white")
        ax.axvline(x = 0, color = "black", linewidth = 0.8, linestyle = "--")
        ax.set_title(col, fontsize = 12, weight = "bold")
        ax.set_xlabel("회귀계수")

    plt.suptitle("OLS / Ridge / Lasso 회귀계수 비교\n(빨강: 양수 / 색상: 음수 / Lasso 0값: 제거된 변수)",
                 fontsize = 12, weight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regression_regularized_comparison.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[정규화 회귀 비교] 저장 완료")


def print_summary(model):
    """핵심 지표 요약 출력"""
    print("\n" + "=" * 50)
    print("[회귀분석 핵심 지표 요약]")
    print("=" * 50)
    print(f"  R²           : {model.rsquared:.3f}  → 모델이 감정지수 분산의 {model.rsquared * 100:.1f}%를 설명")
    print(f"  Adj. R²      : {model.rsquared_adj:.3f}")
    print(f"  F-통계량 p값  : {model.f_pvalue:.4f}  {'→ 모델 전체 유의미' if model.f_pvalue < 0.05 else '→ 모델 전체 유의미하지 않음'}")
    print(f"  AIC          : {model.aic:.2f}")

    sig_vars = model.pvalues[1:][model.pvalues[1:] < 0.05]
    if not sig_vars.empty:
        print(f"\n  유의미한 변수 (p < 0.05): {sig_vars.index.tolist()}")
    else:
        print("\n  유의미한 변수 없음 (데이터 부족 가능성)")
    print("=" * 50)

if __name__ == "__main__":
    set_universal_font()

    parser = argparse.ArgumentParser(description = "OLS 다중 선형 회귀분석 스크립트")
    parser.add_argument(
        "--input",
        type = str,
        default = "notion_brain_dump_raw_cleaned_impute_standard.csv",
        help = "입력 CSV 파일 경로 (기본값: notion_brain_dump_raw_cleaned_impute_standard.csv / 회귀분석은 standard 스케일링 권장)"
    )
    args = parser.parse_args()

    try:
        df, X, y, feature_cols = load_data(args.input)
        vif_df = check_vif(X)
        model = run_ols(X, y)
        plot_coefficients(model, feature_cols)
        plot_residuals(model, X, y)
        plot_actual_vs_predicted(model, y)
        print_summary(model)

        # VIF > 10 변수가 있으면 자동으로 Ridge / Lasso 실행
        if (vif_df["VIF"] > 10).any():
            print("\n[다중공선성 감지] Ridge / Lasso 정규화 회귀분석을 추가 실행합니다.")
            ridge_model, lasso_model, ridge_r2, lasso_r2 = run_ridge_lasso(X, y)
            plot_regularized_coefficients(model, ridge_model, lasso_model, feature_cols)
            print(f"\n[모델 R² 비교]")
            print(f"  OLS   : {model.rsquared:.3f}")
            print(f"  Ridge : {ridge_r2:.3f}")
            print(f"  Lasso : {lasso_r2:.3f}")
        else:
            print("\n[VIF 양호] 다중공선성 문제 없음, OLS 결과를 사용합니다.")

        print(f"\n[최종 완료] 회귀분석 시각화가 outputs/ 폴더에 저장되었습니다.")
    except FileNotFoundError:
        print(f"[오류] 파일이 없습니다: {args.input}")
        print("전처리 단계를 먼저 실행하거나 --input 경로를 확인해주세요.")
        print("회귀분석은 standard 스케일링 적용 파일 사용을 권장합니다.")
        print("예: python data_preprocessing.py --mode impute --outlier detect --scaler standard")