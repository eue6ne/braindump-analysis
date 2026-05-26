import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 텍스트 마이닝 대상 컬럼 (전처리/분석에서 제외, 별도 처리)
TEXT_COLS = ["브레인 덤프", "오늘의 요약"]

# 스케일링에서 제외할 컬럼 (종속변수는 스케일링하지 않음)
SCALE_EXCLUDE_COLS = ["감정지수"]

def load_and_filter_target(file_path):
    print("원천 데이터 파일 로드 중...")
    df = pd.read_csv(file_path)

    # 행 전체가 비어있거나 핵심 요약이 없는 행 제거
    if "오늘의 요약" in df.columns:
        df = df.dropna(subset = ["오늘의 요약"])

    # 종속변수인 '감정지수'가 비어있는 행 제거
    if "감정지수" in df.columns:
        before_target = len(df)
        df = df.dropna(subset = ["감정지수"])
        after_target = len(df)

        if before_target != after_target:
            print(f"[알림] '감정지수' 컬럼에서 결측치가 발견되어 {before_target - after_target}개의 행을 제거했습니다.")
    else:
        print("[경고] 데이터베이스에 '감정지수' 컬럼이 존재하지 않습니다.")

    # 날짜 컬럼 처리 및 인덱스 설정
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"])
        df = df.sort_values(by = "날짜")
        df = df.set_index("날짜")

    return df

def is_multi_select(series):
    if series.dtype != "object":
        return False
    # na=False: NaN 값에 대해 False를 반환하여 경고 방지
    return series.str.contains(",", na = False).any()

def get_col_types(df):
    """컬럼을 다중선택 / 텍스트 / 범주형 / 수치형으로 분류하여 반환"""
    multi_select_cols = [col for col in df.columns if df[col].astype(str).str.contains(',').any()]
    text_cols = [col for col in df.columns if col in TEXT_COLS]
    exclude_cols = set(multi_select_cols + text_cols)

    remaining_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[remaining_cols].select_dtypes(include = [np.number]).columns.tolist()
    categorical_cols = df[remaining_cols].select_dtypes(exclude = [np.number]).columns.tolist()

    return multi_select_cols, text_cols, numeric_cols, categorical_cols

def handle_drop(df):
    multi_select_cols, text_cols, numeric_cols, categorical_cols = get_col_types(df)

    # 다중선택 컬럼: 결측치를 빈 문자열로 채운 뒤 드롭 대상에서 제외
    for col in multi_select_cols:
        df[col] = df[col].fillna("")

    before_count = len(df)
    # 수치형 + 범주형만 드롭 대상 (다중선택, 텍스트 제외)
    drop_target_cols = numeric_cols + categorical_cols
    df = df.dropna(subset = drop_target_cols)
    after_count = len(df)
    print(f"[제거 모드] 나머지 독립변수 결측치 행 제거 완료: {before_count}개 행 -> {after_count}개 행 남음")
    return df

def handle_impute(df):
    multi_select_cols, text_cols, numeric_cols, categorical_cols = get_col_types(df)

    # 다중선택 컬럼: 아무것도 선택하지 않은 날로 간주 → 빈 문자열로 대체
    for col in multi_select_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("")
            print(f"[대체 모드] '{col}' 다중선택 결측치 → 빈 문자열로 대체")

    # 범주형 독립변수: 최빈값 활용
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_series = df[col].mode()
            fill_val = mode_series[0] if not mode_series.empty else "미지정"
            df[col] = df[col].fillna(fill_val)

    # 수치형 독립변수: scikit-learn의 KNNImputer 활용
    # 결측치가 있는 수치형 컬럼이 존재하고, 샘플이 최소 1개 이상 있을 때 작동
    has_missing_numeric = any(df[col].isnull().sum() > 0 for col in numeric_cols)

    if numeric_cols and has_missing_numeric and len(df) > 0:
        # 이웃의 개수(n_neighbors)는 데이터셋이 작으므로 2~3개로 설정 (기본값 5는 데이터가 적으면 에러 유발 가능)
        n_neighbors = min(3, len(df))
        imputer = KNNImputer(n_neighbors = n_neighbors)

        # imputer는 numpy array를 반환하므로 다시 데이터프레임으로 변환하며 인덱스와 컬럼명 복구
        imputed_array = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_array, columns = numeric_cols, index = df.index)

        for col in numeric_cols:
            df[col] = imputed_numeric_df[col]

    print("[대체 모드] 나머지 독립변수들의 결측치 자동 대체 완료 (범주형: 최빈값 / 수치형: KNN)!")
    return df

def handle_scaling(df, scaler_type):
    """
    수치형 독립변수에 스케일링 적용.
    - standard : 평균 0, 표준편차 1로 변환 (StandardScaler) → 회귀분석, 요인분석, PCA 권장
    - minmax   : 0~1 범위로 변환 (MinMaxScaler)           → 클러스터링, KNN 권장
    - none     : 스케일링 생략                            → 상관계수, 시계열 분석 권장
    """
    if scaler_type == "none":
        print("[스케일링] 스케일링을 생략합니다.")
        return df

    _, _, numeric_cols, _ = get_col_types(df)
    # 종속변수 및 기타 제외 컬럼 제거
    scale_cols = [col for col in numeric_cols if col not in SCALE_EXCLUDE_COLS]

    if not scale_cols:
        print("[스케일링] 스케일링 대상 수치형 컬럼이 없습니다.")
        return df

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()

    scaled_array = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled_array, columns = scale_cols, index = df.index)

    for col in scale_cols:
        df[col] = scaled_df[col]

    print(f"[스케일링] {scaler_type} 스케일링 완료 → 대상 컬럼: {scale_cols}")
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "노션 브레인 덤프 데이터 전처리 스크립트")

    parser.add_argument(
        "--mode",
        type = str,
        default = "drop",
        choices = ["drop", "impute"],
        help = "결측치 처리 방식 선택 (drop: 행 삭제, impute: 독립변수 대체)"
    )
    parser.add_argument(
        "--scaler",
        type = str,
        default = "none",
        choices = ["standard", "minmax", "none"],
        help = "스케일링 방식 선택 (standard: 표준화, minmax: 0~1 정규화, none: 생략 / 기본값: none)"
    )

    args = parser.parse_args()
    raw_csv_path = "notion_brain_dump_raw.csv"

    try:
        raw_df = load_and_filter_target(raw_csv_path)

        if args.mode == "drop":
            cleaned_df = handle_drop(raw_df)
            output_filename = f"notion_brain_dump_cleaned_drop_{args.scaler}.csv"
        elif args.mode == "impute":
            cleaned_df = handle_impute(raw_df)
            output_filename = f"notion_brain_dump_cleaned_impute_{args.scaler}.csv"

        cleaned_df = handle_scaling(cleaned_df, args.scaler)

        print("\n--- 전처리된 데이터프레임 구조 확인 ---")
        print(cleaned_df.head())

        cleaned_df.to_csv(output_filename, encoding = "utf-8-sig")
        print(f"\n'{output_filename}' 파일로 저장 완료!")

    except FileNotFoundError:
        print(f"에러: {raw_csv_path} 파일이 없습니다. notion_loader.py를 먼저 실행해주세요.")