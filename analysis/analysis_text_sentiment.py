# 감정 사전 출처: KNU 한국어 감성사전 (KnuSentiLex) - 울산대학교 자연언어처리연구실
# https://github.com/park1200656/KnuSentiLex

import warnings
warnings.filterwarnings("ignore")

import os
import json
import platform
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from konlpy.tag import Okt

# 시각화 이미지 저장 경로
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok = True)

TARGET_COL = "감정지수"
TEXT_COL   = "브레인 덤프"

# KNU 감정 사전 경로
DEFAULT_LEXICON_PATH = "data/KnuSentiLex/data/SentiWord_info.json"

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

def load_lexicon(lexicon_path):
    """
    KNU 한국어 감성사전 로드.
    polarity: -2(매우 부정) / -1(부정) / 0(중립) / 1(긍정) / 2(매우 긍정)
    word_root 기준으로 사전 구성 (형태소 분석 결과와 매칭하기 위함)
    """
    if not os.path.exists(lexicon_path):
        raise FileNotFoundError(
            f"[오류] 감정 사전 파일을 찾을 수 없습니다: {lexicon_path}\n"
            "아래 명령어로 KNU 감성사전을 다운로드하세요.\n"
            "git clone https://github.com/park1200656/KnuSentiLex.git"
        )

    with open(lexicon_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    # word_root 기준으로 딕셔너리 구성 (중복 시 절댓값이 큰 값 우선)
    lexicon = {}
    for entry in data:
        word  = entry.get("word_root", "").strip()
        score = int(entry.get("polarity", 0))
        if word and (word not in lexicon or abs(score) > abs(lexicon[word])):
            lexicon[word] = score

    print(f"[감정 사전] {len(lexicon)}개 단어 로드 완료 ({lexicon_path})")
    return lexicon

def load_data(file_path):
    """전처리된 CSV 로드"""
    df = pd.read_csv(file_path, index_col = "날짜", parse_dates = True).sort_index()

    if TEXT_COL not in df.columns:
        raise ValueError(f"[오류] '{TEXT_COL}' 컬럼이 없습니다.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"[오류] '{TARGET_COL}' 컬럼이 없습니다.")

    df = df.dropna(subset = [TEXT_COL])
    print(f"[데이터 로드] {len(df)}행 / 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df

def compute_sentiment_score(text, okt, lexicon):
    """
    텍스트의 KNU 감정 점수 계산.
    형태소 분석 후 감정 사전과 매칭하여 단어별 점수 합산.
    점수 범위: -2 ~ 2 (단어당) → 텍스트 전체 평균으로 정규화
    매칭 단어가 없으면 0 반환.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    morphs = okt.morphs(text, norm = True, stem = True)
    scores = [lexicon[w] for w in morphs if w in lexicon]

    return float(np.mean(scores)) if scores else 0.0

def add_sentiment_scores(df, lexicon):
    """전체 데이터프레임에 일별 KNU 감정 점수 컬럼 추가"""
    print("\n[감정 점수 계산] 형태소 분석 중... (시간이 걸릴 수 있습니다)")
    okt = Okt()
    df["감정점수_NLP"] = df[TEXT_COL].apply(lambda t: compute_sentiment_score(t, okt, lexicon))

    pos = (df["감정점수_NLP"] > 0).sum()
    neg = (df["감정점수_NLP"] < 0).sum()
    neu = (df["감정점수_NLP"] == 0).sum()
    print(f"  긍정: {pos}일 / 부정: {neg}일 / 중립: {neu}일")
    print(f"  평균 감정 점수: {df['감정점수_NLP'].mean():.3f}")
    return df

def plot_sentiment_trend(df):
    """
    일별 NLP 감정 점수 추이 + 이동평균 시각화.
    - 양수(빨강): 긍정적인 텍스트
    - 음수(파랑): 부정적인 텍스트
    - 7일 이동평균으로 전반적인 흐름 파악
    """
    ma7 = df["감정점수_NLP"].rolling(window = 7, center = True).mean()

    fig, ax = plt.subplots(figsize = (12, 5))

    # 긍정/부정 구간 색상 채우기
    ax.fill_between(df.index, df["감정점수_NLP"], 0,
                    where = df["감정점수_NLP"] >= 0,
                    color = "crimson", alpha = 0.3, label = "긍정 구간")
    ax.fill_between(df.index, df["감정점수_NLP"], 0,
                    where = df["감정점수_NLP"] < 0,
                    color = "steelblue", alpha = 0.3, label = "부정 구간")

    ax.plot(df.index, df["감정점수_NLP"], color = "gray", linewidth = 0.8, alpha = 0.6)
    ax.plot(ma7.index, ma7.values, color = "black", linewidth = 2.0, label = "7일 이동평균")
    ax.axhline(y = 0, color = "black", linestyle = "--", linewidth = 0.8)

    ax.set_title("일별 NLP 감정 점수 추이\n(KNU 감성사전 기반 / 빨강: 긍정 / 파랑: 부정)", fontsize = 12, weight = "bold")
    ax.set_ylabel("감정 점수 (-2 ~ 2)")
    ax.legend(fontsize = 9)
    ax.grid(True, linestyle = ":", alpha = 0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation = 35)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_trend.png", dpi = 300)
    plt.close()
    print("[NLP 감정 점수 추이] 저장 완료")

def plot_sentiment_vs_score(df):
    """
    NLP 감정 점수 vs 자기보고 감정지수 산점도 + 상관계수.
    - 두 지표가 일치할수록 점들이 우상향으로 분포
    - 괴리가 큰 날: 감정을 실제와 다르게 표현했을 가능성
    """
    corr = df["감정점수_NLP"].corr(df[TARGET_COL])

    plt.figure(figsize = (7, 6))
    plt.scatter(df["감정점수_NLP"], df[TARGET_COL],
                alpha = 0.5, color = "steelblue", s = 40, edgecolors = "white")

    # 추세선
    z = np.polyfit(df["감정점수_NLP"], df[TARGET_COL], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["감정점수_NLP"].min(), df["감정점수_NLP"].max(), 100)
    plt.plot(x_line, p(x_line), color = "crimson", linewidth = 1.5, linestyle = "--", label = "추세선")

    plt.axvline(x = 0, color = "gray", linestyle = ":", linewidth = 0.8)
    plt.axhline(y = df[TARGET_COL].mean(), color = "gray", linestyle = ":", linewidth = 0.8)

    plt.xlabel("NLP 감정 점수 (KNU 사전 기반)")
    plt.ylabel("자기보고 감정지수 (1~10)")
    plt.title(
        f"NLP 감정 점수 vs 자기보고 감정지수\n상관계수: {corr:.3f}",
        fontsize = 12, weight = "bold"
    )
    plt.legend(fontsize = 9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_vs_score.png", dpi = 300)
    plt.close()
    print(f"[NLP vs 자기보고 산점도] 저장 완료 (상관계수: {corr:.3f})")

def plot_sentiment_gap(df, top_n = 10):
    """
    NLP 감정 점수와 자기보고 감정지수의 괴리가 큰 날 시각화.
    NLP 점수를 1~10 스케일로 변환 후 차이 계산.
    괴리가 클수록 텍스트 표현과 실제 감정이 다른 날.
    """
    # NLP 점수를 1~10 스케일로 변환 (-2~2 → 1~10)
    nlp_scaled = (df["감정점수_NLP"] + 2) / 4 * 9 + 1
    gap = (nlp_scaled - df[TARGET_COL]).abs()

    gap_df = pd.DataFrame({"날짜": df.index.strftime("%Y-%m-%d"),
                           "자기보고": df[TARGET_COL].values,
                           "NLP점수(변환)": nlp_scaled.values,
                           "괴리": gap.values}).sort_values("괴리", ascending = False).head(top_n)

    fig, ax = plt.subplots(figsize = (10, max(5, top_n * 0.5)))
    x = np.arange(len(gap_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, gap_df["자기보고"], width, label = "자기보고 감정지수", color = "steelblue", alpha = 0.8)
    bars2 = ax.bar(x + width / 2, gap_df["NLP점수(변환)"], width, label = "NLP 감정 점수 (1~10 변환)", color = "darkorange", alpha = 0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(gap_df["날짜"], rotation = 35, ha = "right", fontsize = 9)
    ax.set_ylabel("감정 점수 (1~10)")
    ax.set_ylim(0, 11)
    ax.legend(fontsize = 9)
    ax.set_title(f"자기보고 감정지수와 NLP 감정 점수 괴리 상위 {top_n}일\n(두 막대 차이가 클수록 텍스트 표현과 실제 감정이 다른 날)",fontsize = 12, weight = "bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_gap.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[감정 괴리 상위 {top_n}일] 저장 완료")

def print_summary(df):
    """핵심 지표 요약 출력"""
    corr = df["감정점수_NLP"].corr(df[TARGET_COL])
    nlp_scaled = (df["감정점수_NLP"] + 2) / 4 * 9 + 1
    gap = (nlp_scaled - df[TARGET_COL]).abs()

    print("\n" + "=" * 55)
    print("[NLP 감정 분석 요약]")
    print("=" * 55)
    print(f"  NLP 감정 점수 평균  : {df['감정점수_NLP'].mean():.3f} (-2 ~ 2)")
    print(f"  자기보고 감정지수 평균: {df[TARGET_COL].mean():.2f} (1 ~ 10)")
    print(f"  두 지표 상관계수     : {corr:.3f}")
    if corr >= 0.5:
        print("  → 텍스트 감정과 자기보고 감정이 비교적 일치합니다.")
    elif corr >= 0.3:
        print("  → 텍스트 감정과 자기보고 감정이 어느 정도 일치합니다.")
    else:
        print("  → 텍스트 표현과 실제 감정 사이에 괴리가 있을 수 있습니다.")
    print(f"  평균 괴리 (1~10 기준): {gap.mean():.2f}점")
    print(f"  최대 괴리 날짜       : {df.index[gap.argmax()].date()} ({gap.max():.2f}점)")
    print("=" * 55)

if __name__ == "__main__":
    set_universal_font()

    parser = argparse.ArgumentParser(description = "KNU 감성사전 기반 NLP 감정 분석 스크립트")
    parser.add_argument(
        "--input",
        type = str,
        default = "notion_brain_dump_raw_cleaned_impute_none.csv",
        help = "입력 CSV 파일 경로 (기본값: notion_brain_dump_raw_cleaned_impute_none.csv)"
    )
    parser.add_argument(
        "--lexicon",
        type = str,
        default = DEFAULT_LEXICON_PATH,
        help = f"KNU 감성사전 JSON 파일 경로 (기본값: {DEFAULT_LEXICON_PATH})"
    )
    parser.add_argument(
        "--gap_top",
        type = int,
        default = 10,
        help = "감정 괴리 상위 N일 표시 (기본값: 10)"
    )
    args = parser.parse_args()

    try:
        lexicon = load_lexicon(args.lexicon)
        df = load_data(args.input)
        df = add_sentiment_scores(df, lexicon)

        plot_sentiment_trend(df)
        plot_sentiment_vs_score(df)
        plot_sentiment_gap(df, top_n = args.gap_top)
        print_summary(df)

        print(f"\n[최종 완료] NLP 감정 분석 시각화가 outputs/ 폴더에 저장되었습니다.")

    except (FileNotFoundError, ValueError) as e:
        print(e)
        print("전처리 단계를 먼저 실행하거나 --input 경로를 확인해주세요.")