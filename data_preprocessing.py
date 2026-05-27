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
    multi_select_cols = [col for col in df.columns if df[col].dropna().astype(str).str.contains(',').any()]
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

def handle_outlier(df, outlier_type):
    """
    수치형 독립변수의 이상치를 IQR 방식으로 탐지 후 사용자 확인을 거쳐 처리.
    - detect : 이상치를 탐지하고 내용을 출력한 뒤, 처리 방식을 대화형으로 선택
    - none   : 이상치 처리 생략 (기본값)

    감정일기 특성상 이상치가 실제로 의미 있는 날일 수 있으므로 (예: 감정지수 1점인 최악의 날)
    탐지 결과를 확인 후 처리 여부를 직접 결정하는 것을 권장.
    """
    if outlier_type == "none":
        print("[이상치] 이상치 처리를 생략합니다.")
        return df

    _, _, numeric_cols, _ = get_col_types(df)
    # 종속변수 제외
    target_cols = [col for col in numeric_cols if col not in SCALE_EXCLUDE_COLS]

    if not target_cols:
        print("[이상치] 이상치 탐지 대상 수치형 컬럼이 없습니다.")
        return df

    # 1단계: 전체 이상치 탐지
    outlier_info = {}  # {col: (is_outlier, lower, upper)}

    for col in target_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_outlier = (df[col] < lower) | (df[col] > upper)

        if is_outlier.sum() > 0:
            outlier_info[col] = (is_outlier, lower, upper)

    if not outlier_info:
        print("[이상치] 탐지된 이상치가 없습니다.")
        return df

    # 2단계: 탐지 결과 출력
    print("\n" + "=" * 50)
    print("[이상치] 이상치로 감지된 데이터는 아래와 같습니다.")
    print("=" * 50)

    all_outlier_flags = pd.Series(False, index = df.index)

    for col, (is_outlier, lower, upper) in outlier_info.items():
        outlier_rows = df[is_outlier][[col]]
        all_outlier_flags = all_outlier_flags | is_outlier
        print(f"\n▶ [{col}] 정상 범위: {lower:.2f} ~ {upper:.2f} | 이상치 {is_outlier.sum()}개")
        print(outlier_rows.to_string())

    print("\n" + "=" * 50)
    print(f"총 {all_outlier_flags.sum()}개 행에서 이상치가 탐지되었습니다.")
    print("=" * 50)

    # 3단계: 처리 여부 확인
    answer = input("\n이상치 처리를 원하시나요? [y/n]: ").strip().lower()

    if answer != "y":
        print("[이상치] 이상치를 처리하지 않고 원본 값을 유지합니다.")
        return df

    # 4단계: 처리 방식 선택
    print("\n처리 방식을 선택해주세요.")
    print("  [1] flag — 이상치 행에 '이상치_플래그' 컬럼 추가 (값 보존, 분석 시 참고용)")
    print("  [2] cap  — 이상치를 IQR 경계값으로 대체 (Winsorizing)")
    method = input("선택 [1/2]: ").strip()

    if method not in ["1", "2"]:
        print("[이상치] 올바른 선택이 아닙니다. 이상치를 처리하지 않고 원본 값을 유지합니다.")
        return df

    # 5단계: 컬럼별 처리할 행 선택
    selected_indices = {}  # {col: [선택된 index 리스트]}

    for col, (is_outlier, lower, upper) in outlier_info.items():
        outlier_dates = df[is_outlier].index.tolist()
        outlier_vals  = df[is_outlier][col].tolist()

        print(f"\n▶ [{col}] 에서 이상치 {len(outlier_dates)}개가 탐지되었습니다.")
        print(f"   아래 목록에서 처리할 항목의 번호를 입력하세요.")
        print(f"   (번호는 이상치 목록 기준이며, 전체 데이터 순서와 다를 수 있습니다.)")
        print(f"   (입력 예시: 1 → 1번만 처리 / 1,3 → 1번과 3번 처리 / all → 전체 처리)")
        for idx, (date, val) in enumerate(zip(outlier_dates, outlier_vals), start = 1):
            print(f"  {idx}. {str(date)[:10]} | {val}")

        row_input = input("\n입력: ").strip().lower()

        if row_input == "all":
            selected_indices[col] = outlier_dates
        else:
            try:
                chosen = [int(x.strip()) for x in row_input.split(",")]
                selected_indices[col] = [outlier_dates[i - 1] for i in chosen if 1 <= i <= len(outlier_dates)]
            except (ValueError, IndexError):
                print(f"  [경고] 올바른 입력이 아닙니다. [{col}] 이상치 처리를 건너뜁니다.")
                selected_indices[col] = []

    # 6단계: 선택된 행에 처리 적용
    flag_indices = []

    for col, indices in selected_indices.items():
        if not indices:
            continue
        _, lower, upper = outlier_info[col]

        if method == "1":
            flag_indices.extend(indices)

        elif method == "2":
            for idx in indices:
                original = df.at[idx, col]
                df.at[idx, col] = max(lower, min(upper, original))
            print(f"[이상치] '{col}' {len(indices)}개 행 → 경계값으로 대체 완료 ({lower:.2f} ~ {upper:.2f})")

    if method == "1" and flag_indices:
        unique_flags = list(dict.fromkeys(flag_indices))  # 중복 제거, 순서 유지
        df["이상치_플래그"] = df.index.isin(unique_flags).astype(int)
        print(f"[이상치] '이상치_플래그' 컬럼 추가 완료 ({len(unique_flags)}개 행 표시)")

    return df


def handle_encode_multiselect(df):
    """
    다중선택(Multi-select) 컬럼을 항목별 0/1 이진 컬럼으로 분리 (MLB 인코딩).
    전처리 파이프라인에서 항상 실행됨.

    예시)
      여가_취미유형: "애니, 독서"
      → 여가_취미유형_애니: 1 / 여가_취미유형_독서: 1 / 여가_취미유형_게임: 0
    """
    multi_select_cols, _, _, _ = get_col_types(df)
    # TEXT_COLS는 MLB 인코딩 대상에서 제외
    multi_select_cols = [col for col in multi_select_cols if col not in TEXT_COLS]

    if not multi_select_cols:
        print("[인코딩] 다중선택 컬럼이 없습니다.")
        return df

    for col in multi_select_cols:
        # 빈 문자열 및 NaN 처리 후 항목 분리
        unique_items = set()
        for val in df[col].dropna():
            if val == "":
                continue
            for item in str(val).split(","):
                unique_items.add(item.strip())

        # 항목별 0/1 이진 컬럼 생성
        for item in sorted(unique_items):
            col_name = f"{col}_{item}"
            df[col_name] = df[col].apply(
                lambda val: 1 if isinstance(val, str) and item in [v.strip() for v in val.split(",")] else 0
            )

        # 원본 다중선택 컬럼 제거
        df = df.drop(columns = [col])
        print(f"[인코딩] '{col}' → {len(unique_items)}개 이진 컬럼으로 분리 완료: {sorted(unique_items)}")

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
    parser.add_argument(
        "--outlier",
        type = str,
        default = "none",
        choices = ["detect", "none"],
        help = "이상치 처리 방식 선택 (detect: 탐지 후 대화형 처리, none: 생략 / 기본값: none)"
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

        cleaned_df = handle_outlier(cleaned_df, args.outlier)
        cleaned_df = handle_encode_multiselect(cleaned_df)
        cleaned_df = handle_scaling(cleaned_df, args.scaler)

        print("\n--- 전처리된 데이터프레임 구조 확인 ---")
        print(cleaned_df.head())

        cleaned_df.to_csv(output_filename, encoding = "utf-8-sig")
        print(f"\n'{output_filename}' 파일로 저장 완료!")

    except FileNotFoundError:
        print(f"에러: {raw_csv_path} 파일이 없습니다. notion_loader.py를 먼저 실행해주세요.")