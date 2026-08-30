# 브레인 덤프 기반 개인 감정 데이터 분석 시스템

Notion API를 활용하여 매일 기록한 브레인 덤프 데이터를 수집하고, 전처리 및 통계 분석을 수행하는 개인 데이터 분석 프로젝트입니다.


---

## 1. Project Objective

본 프로젝트는 매일의 감정, 신체 컨디션, 업무 패턴, 여가 활동 등을 노션 데이터베이스에 기록하고, 이를 Python으로 분석하여 감정지수에 영향을 미치는 요인을 파악하는 것을 목표로 합니다.

단순 일기를 넘어,

- 데이터 수집 및 전처리 파이프라인 설계
- 탐색적 데이터 분석 (EDA)
- 요인분석 / 회귀분석 / 클러스터링 / 시계열 분석 / 텍스트 마이닝

의 전 과정을 직접 구축하였으며, 사용자가 본인 맞춤형 변수를 자유롭게 추가할 수 있는 범용적인 구조로 설계하였습니다.

---

## 2. Dataset

- Source: 개인 Notion 데이터베이스 (Notion API 연동)
- 수집 주기: 매일 1회 기록
- 주요 변수 예시:
  - 감정지수 (1~10, 종속변수)
  - 신체 컨디션 (수면시간, 수면의질, 카페인 섭취량 등)
  - 업무 패턴 (업무강도, 야근여부, 동료관계 등)
  - 여가 활동 (취미유형, 취미만족도 등)
  - 환경 (날씨 등)
  - 브레인 덤프 (자유 기술 일기, 텍스트 마이닝 대상)

> 실제 데이터는 개인정보 보호를 위해 공개하지 않습니다.
> 코드 재현을 위한 샘플 데이터(`data/sample_data.csv`)를 제공합니다.

---

## 3. Sample Data

`data/sample_data.csv`는 코드 재현용 가상 데이터입니다.
입사 3개월차 직장인 적응기 시나리오를 직접 설계하고 Claude의 도움을 받아 생성했습니다.
실제 분석 결과는 개인 기록 데이터를 기반으로 도출됩니다.

| 구간 | 기간 | 특징 |
|------|------|------|
| 초반 긴장기 | 1~2주 | 낯선 환경, 감정 불안정 |
| 적응 위기 | 3~5주 | 실수, 눈치, 감정지수 하락 |
| 안정기 | 6~9주 | 업무 적응, 감정지수 상승 |
| 번아웃 | 10~12주 | 야근 증가, 무기력, 급격한 하락 |
| 회복기 | 13주~ | 서서히 회복 |

---

## 4. Methodology

1. Notion API를 통한 데이터 수집
2. 결측치 처리 (최빈값 / KNN Imputer) 및 이상치 탐지 (IQR)
3. 다중선택 컬럼 MLB 인코딩 및 스케일링
4. 탐색적 데이터 분석 및 시각화
5. 요인분석 (PCA) — 변수 구조 파악
6. 회귀분석 (OLS + Ridge/Lasso) — 감정지수 예측 모델
7. 클러스터링 (K-Means) — 하루 패턴 유형화
8. 시계열 분석 — 감정지수 추세 및 주기성
9. 텍스트 마이닝 (KoNLPy) — 브레인 덤프 키워드 분석
10. NLP 감정 분석 (KNU 감성사전) — 텍스트 감정 점수 산출 및 자기보고 감정지수 비교 (보조 분석)

---

## 5. Key Findings

직접 기록한 100일간의 실제 데이터를 바탕으로 분석 진행하였습니다. 자세한 내용은 **[FINDINGS.md](FINDINGS.md)** 에서 확인하실 수 있습니다.

---

## 6. Limitations

본 분석은 개인 데이터 기반의 탐색적 분석이며, 표본 크기 및 개인 특성에 따라 결과의 일반화에 한계가 있습니다.
통계적 유의성보다는 개인 패턴 파악 및 생활 습관 개선에 초점을 둔 프로젝트입니다.

---

## 7. Repository Structure

```
braindump-analysis/
│
├── analysis/
│   ├── analysis_clustering.py     # K-Means 클러스터링
│   ├── analysis_factor.py         # PCA 요인분석
│   ├── analysis_regression.py     # OLS / Ridge / Lasso 회귀분석
│   ├── analysis_text.py           # 텍스트 마이닝
│   ├── analysis_text_sentiment.py # NLP 감정 분석 (보조 분석)
│   ├── analysis_timeseries.py     # 시계열 분석
│   └── stopwords.txt              # 텍스트 마이닝 사용자 정의 불용어
│
├── data/
│   └── sample_data.csv            # 코드 재현용 샘플 데이터
│
├── findings_images/
│   └── *.png                      # key findings에서 사용한 이미지    
│
├── outputs/                       # 시각화 결과 저장
│
├── pages/                         # Streamlit을 이용한 웹앱 분석별 파일
│   └── 1_데이터수집및전처리.py
│   └── 2_EDA.py
│   └── 3_요인분석.py
│   └── 4_회귀분석.py
│   └── 5_클러스터링.py
│   └── 6_시계열분석.py
│   └── 7_텍스트마이닝.py
│
├── .gitignore
├── app.py                         # Streamlit을 이용한 웹앱
├── data_preprocessing.py          # 결측치 / 이상치 / 인코딩 / 스케일링
├── data_visualization.py          # EDA 시각화
├── FINDINGS.md                    # key findings
├── GUIDELINE.md                   # 노션 DB 설정 및 분석 해석 가이드
├── notion_loader.py               # Notion API 데이터 수집
├── README.md
└── requirements.txt
```

※ `outputs/`, 실제 데이터 CSV 파일, `data/KnuSentiLex/`는 Git에 포함하지 않습니다.

---

## 8. How to Run

### Environment
- Python 3.13
- Java (KoNLPy 의존성, macOS 기준 `brew install openjdk`)

### 1. Clone repository

```bash
git clone https://github.com/eue6ne/braindump-analysis.git
cd braindump-analysis
```

### 2. Create virtual environment

```bash
python -m venv braindump-analysis-venv
source braindump-analysis-venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. KNU 감성사전 다운로드 (NLP 감정 분석 실행 시 필요)

```bash
git clone https://github.com/park1200656/KnuSentiLex.git data/KnuSentiLex
```

### 5. 실제 데이터로 실행 (Notion 연동)
`.env` 파일을 프로젝트 루트에 생성합니다. 해당 파일에 노션 토큰과 노션 데이터베이스 ID를 입력하여 저장합니다.

```
NOTION_TOKEN=your_notion_api_token
NOTION_DATABASE_ID=your_database_id
```

```bash
# 데이터 수집
python notion_loader.py

# 전처리
python data_preprocessing.py --mode impute --outlier detect --scaler none

# EDA 시각화
python data_visualization.py

# 분석
python analysis/analysis_factor.py
python analysis/analysis_regression.py --input notion_brain_dump_raw_cleaned_impute_standard.csv
python analysis/analysis_clustering.py --input notion_brain_dump_raw_cleaned_impute_minmax.csv --exclude_mlb
python analysis/analysis_timeseries.py
python analysis/analysis_text.py

# NLP 감정 분석 (보조 분석, KnuSentiLex 다운로드 필요)
python analysis/analysis_text_sentiment.py
```

### 6. 샘플 데이터로 실행

```bash
# 전처리
python data_preprocessing.py --mode impute --outlier none --scaler none --input data/sample_data.csv

# EDA 시각화
python data_visualization.py --input sample_data_cleaned_impute_none.csv

# 분석
python analysis/analysis_factor.py --input sample_data_cleaned_impute_none.csv
python analysis/analysis_regression.py --input sample_data_cleaned_impute_standard.csv
python analysis/analysis_clustering.py --input sample_data_cleaned_impute_minmax.csv --exclude_mlb
python analysis/analysis_timeseries.py --input sample_data_cleaned_impute_none.csv
python analysis/analysis_text.py --input sample_data_cleaned_impute_none.csv

# NLP 감정 분석 (보조 분석, KnuSentiLex 다운로드 필요)
python analysis/analysis_text_sentiment.py --input sample_data_cleaned_impute_none.csv
```

> 노션 DB 설정, 분석 해석 기준, 스케일링 옵션 선택 등 자세한 내용은 **[GUIDELINE.md](GUIDELINE.md)** 를 참고하세요.

---

## 9. Author

Personal Data Analysis Project    
2026