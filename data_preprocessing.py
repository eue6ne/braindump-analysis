import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

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
            pass
    else:
        print("[경고] 데이터베이스에 '감정지수' 컬럼이 존재하지 않습니다.")
    
    # 날짜 컬럼 처리 및 인덱스 설정
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"])
        df = df.sort_values(by = "날짜")
        df = df.set_index("날짜")
        
    return df

def handle_drop(df):
    before_count = len(df)
    df = df.dropna()
    after_count = len(df)
    print(f"[제거 모드] 나머지 독립변수 결측치 행 제거 완료: {before_count}개 행 -> {after_count}개 행 남음")
    return df

def is_multi_select(series):
    if series.dtype != 'object':
        return False
    
    # 쉼표(,)가 포함되어 있는지 확인 (다중 선택(Multi-select) 타입의 핵심 특징)
    return series.str.contains(',').any()

def handle_impute(df):
    # 다중 선택(Multi-select) 타입 추출
    multi_select_cols = [col for col in df.columns if is_multi_select(df[col])]
    
    remaining_cols = df.columns.drop(multi_select_cols)
    numeric_cols = df[remaining_cols].select_dtypes(include = [np.number]).columns.tolist()
    categorical_cols = df[remaining_cols].select_dtypes(exclude = [np.number]).columns.tolist()
    
    # 범주형 독립변수: 최빈값 활용
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            # 최빈값이 여러 개일 경우 첫 번째 값을 사용
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
        
        # 원본 데이터프레임에 업데이트
        for col in numeric_cols:
            df[col] = imputed_numeric_df[col]
            
    print("[대체 모드] 나머지 독립변수들의 결측치 자동 대체 완료 (범주형: 최빈값 / 수치형: KNN)!")
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
    
    args = parser.parse_args()
    raw_csv_path = "notion_brain_dump_raw.csv"
    
    try:
        raw_df = load_and_filter_target(raw_csv_path)
        
        if args.mode == "drop":
            cleaned_df = handle_drop(raw_df)
            output_filename = "notion_brain_dump_cleaned_drop.csv"
        elif args.mode == "impute":
            cleaned_df = handle_impute(raw_df)
            output_filename = "notion_brain_dump_cleaned_impute.csv"
            
        print("\n--- 전처리된 데이터프레임 구조 확인 ---")
        print(cleaned_df.head())
        
        cleaned_df.to_csv(output_filename, encoding = "utf-8-sig")
        print(f"\n'{output_filename}' 파일로 저장 완료!")
        
    except FileNotFoundError:
        print(f"에러: {raw_csv_path} 파일이 없습니다. notion_loader.py를 먼저 실행해주세요.")