import warnings
warnings.filterwarnings("ignore")

import os
import platform
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud

# 시각화 이미지 저장 경로
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok = True)

TARGET_COL  = "감정지수"
TEXT_COL    = "브레인 덤프"

# 분석 대상 품사 (명사 + 형용사)
TARGET_POS = ["Noun", "Adjective"]

# 기본 불용어 (도메인 무관, 분석 의미 없는 단어)
DEFAULT_STOPWORDS = {
    "것", "수", "나", "날", "오늘", "하루", "생각", "말", "때",
    "거", "내", "그", "이", "좀", "더", "뭔가", "뭘", "왜",
    "다", "안", "못", "잘", "또", "진짜", "너무", "정말", "어쩔",
    "가", "에", "를", "이다", "있다", "없다", "하다", "되다",
    "같다", "보다", "되다", "같은", "있는", "없는", "하는"
}

def load_stopwords(stopwords_path = "stopwords.txt"):
    """
    외부 불용어 파일 로드.
    stopwords.txt가 없으면 기본 불용어만 사용.
    파일이 있으면 기본 불용어 + 사용자 정의 불용어를 합쳐서 반환.
    stopwords.txt 형식: 한 줄에 단어 하나 (# 으로 시작하는 줄은 주석)
    """
    stopwords = set(DEFAULT_STOPWORDS)

    if not os.path.exists(stopwords_path):
        print(f"[불용어] {stopwords_path} 없음 → 기본 불용어 {len(stopwords)}개 사용")
        return stopwords

    with open(stopwords_path, "r", encoding = "utf-8") as f:
        custom = {
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        }

    stopwords = stopwords | custom
    print(f"[불용어] 기본 {len(DEFAULT_STOPWORDS)}개 + 사용자 정의 {len(custom)}개 = 총 {len(stopwords)}개 사용")
    return stopwords

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

def get_wordcloud_font():
    """워드클라우드용 한글 폰트 경로 반환"""
    os_name = platform.system()
    candidates = {
        "Darwin": [
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/Library/Fonts/NanumGothic.ttf",
        ],
        "Windows": [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/NanumGothic.ttf",
        ],
    }
    for path in candidates.get(os_name, []):
        if os.path.exists(path):
            return path

    # Linux: fonts-nanum 설치 필요
    for path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "[오류] 한글 워드클라우드 폰트를 찾을 수 없습니다.\n"
        "macOS: AppleGothic 기본 내장\n"
        "Linux: sudo apt install fonts-nanum\n"
        "Windows: NanumGothic 설치 권장"
    )

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

def extract_tokens(texts, stopwords):
    """Okt 형태소 분석기로 명사 + 형용사 추출 후 불용어 제거"""
    okt = Okt()
    tokens = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        morphs = okt.pos(text, norm = True, stem = True)
        for word, pos in morphs:
            if pos in TARGET_POS and len(word) > 1 and word not in stopwords:
                tokens.append(word)
    return tokens

def plot_top_keywords(tokens, title, filename, top_n = 20, color = "steelblue"):
    """상위 키워드 빈도 막대그래프"""
    counter = Counter(tokens)
    top = counter.most_common(top_n)
    if not top:
        print(f"[{filename}] 추출된 키워드 없음, 생략")
        return counter

    words, counts = zip(*top)
    plt.figure(figsize = (10, max(5, top_n * 0.4)))
    bars = plt.barh(list(reversed(words)), list(reversed(counts)), color = color, alpha = 0.8, edgecolor = "white")
    plt.title(title, fontsize = 12, weight = "bold")
    plt.xlabel("빈도")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[{filename}] 저장 완료")
    return counter

def plot_wordcloud(tokens, title, filename, font_path):
    """워드클라우드 시각화"""
    if not tokens:
        print(f"[{filename}] 토큰 없음, 생략")
        return

    counter = Counter(tokens)
    wc = WordCloud(font_path = font_path, background_color = "white", width = 800, height = 500, 
                   max_words = 100, colormap = "coolwarm").generate_from_frequencies(counter)

    plt.figure(figsize = (10, 6))
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.title(title, fontsize = 13, weight = "bold", pad = 15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[{filename}] 저장 완료")

def plot_high_low_keywords(df, font_path, stopwords, threshold = 0.3, top_n = 15):
    """
    감정지수 상위 / 하위 구간별 키워드 비교.
    - 상위 threshold: 감정지수 상위 30% (기본값) → 긍정적인 날
    - 하위 threshold: 감정지수 하위 30% (기본값) → 부정적인 날
    두 그룹의 키워드 차이를 통해 감정에 영향을 주는 표현 파악.
    """
    high_q = df[TARGET_COL].quantile(1 - threshold)
    low_q  = df[TARGET_COL].quantile(threshold)

    high_texts = df[df[TARGET_COL] >= high_q][TEXT_COL].tolist()
    low_texts  = df[df[TARGET_COL] <= low_q][TEXT_COL].tolist()

    print(f"\n[상위/하위 비교] 상위 기준: {high_q:.1f}점 이상 ({len(high_texts)}일) / 하위 기준: {low_q:.1f}점 이하 ({len(low_texts)}일)")

    high_tokens = extract_tokens(high_texts, stopwords)
    low_tokens  = extract_tokens(low_texts, stopwords)

    high_counter = Counter(high_tokens)
    low_counter  = Counter(low_tokens)

    # 상위/하위 워드클라우드 비교
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    for ax, counter, title, cmap in zip(axes, [high_counter, low_counter], 
                                        [f"감정지수 상위 {int(threshold * 100)}% 키워드", f"감정지수 하위 {int(threshold * 100)}% 키워드"],
                                        ["Blues", "Reds"]):
        if counter:
            wc = WordCloud(font_path = font_path, background_color = "white", width = 500,  height = 350, 
                           max_words = 60, colormap = cmap).generate_from_frequencies(counter)
            ax.imshow(wc, interpolation = "bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize = 12, weight = "bold")

    plt.suptitle("감정지수 상위 vs 하위 구간 키워드 비교", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/text_highlow_wordcloud.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[상위/하위 워드클라우드] 저장 완료")

    # 상위에만 많이 등장하는 단어 vs 하위에만 많이 등장하는 단어
    all_words = set(high_counter.keys()) | set(low_counter.keys())
    diff_data = []
    for word in all_words:
        h = high_counter.get(word, 0)
        l = low_counter.get(word, 0)
        if h + l >= 3:  # 최소 3회 이상 등장한 단어만
            diff_data.append({"단어": word, "상위빈도": h, "하위빈도": l, "차이": h - l})

    if diff_data:
        diff_df = pd.DataFrame(diff_data).sort_values("차이", ascending = False)
        top_high_words = diff_df.head(top_n)
        top_low_words  = diff_df.tail(top_n).iloc[::-1]

        fig, axes = plt.subplots(1, 2, figsize = (14, max(5, top_n * 0.45)))

        axes[0].barh(top_high_words["단어"], top_high_words["차이"], color = "steelblue", alpha = 0.8, edgecolor = "white")
        axes[0].set_title(f"감정 높은 날에 더 많이 등장한 단어 (상위 {top_n}개)", fontsize = 11, weight = "bold")
        axes[0].set_xlabel("빈도 차이 (상위 - 하위)")
        axes[0].axvline(x = 0, color = "black", linewidth = 0.8, linestyle = "--")

        axes[1].barh(top_low_words["단어"], top_low_words["차이"].abs(), color = "crimson", alpha = 0.8, edgecolor = "white")
        axes[1].set_title(f"감정 낮은 날에 더 많이 등장한 단어 (상위 {top_n}개)", fontsize = 11, weight = "bold")
        axes[1].set_xlabel("빈도 차이 (하위 - 상위)")

        plt.suptitle("감정지수 상위 vs 하위 구간 키워드 차이 분석", fontsize = 13, weight = "bold", y = 1.02)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/text_highlow_diff.png", dpi = 300, bbox_inches = "tight")
        plt.close()
        print("[상위/하위 키워드 차이] 저장 완료")

def plot_keyword_mood(df, tokens_by_row, top_n = 20):
    """
    특정 키워드가 등장한 날의 평균 감정지수 막대그래프.
    등장 횟수가 적은 키워드는 신뢰도가 낮으므로 최소 등장 횟수 필터링.
    """
    # 행별 토큰 딕셔너리 생성
    word_mood = {}
    for idx, tokens in tokens_by_row.items():
        mood = df.loc[idx, TARGET_COL]
        for word in set(tokens):  # 같은 날 중복 제거
            if word not in word_mood:
                word_mood[word] = []
            word_mood[word].append(mood)

    # 최소 3회 이상 등장한 키워드만
    min_count = 3
    mood_data = [{"단어": w, "평균감정지수": np.mean(moods), "등장횟수": len(moods)}
                 for w, moods in word_mood.items() if len(moods) >= min_count]

    if not mood_data:
        print("[키워드별 감정지수] 조건 충족 키워드 없음, 생략")
        return

    mood_df = pd.DataFrame(mood_data).sort_values("평균감정지수", ascending = False)
    top_df = pd.concat([mood_df.head(top_n // 2), mood_df.tail(top_n // 2)]).drop_duplicates()

    overall_mean = df[TARGET_COL].mean()
    bar_colors = ["steelblue" if v >= overall_mean else "crimson" for v in top_df["평균감정지수"]]

    plt.figure(figsize = (9, max(6, len(top_df) * 0.45)))
    bars = plt.barh(top_df["단어"], top_df["평균감정지수"], color = bar_colors, alpha = 0.8, edgecolor = "white")
    plt.axvline(x = overall_mean, color = "black", linestyle = "--", linewidth = 1.2, label = f"전체 평균 ({overall_mean:.1f})")
    plt.title(f"키워드별 평균 감정지수 (등장 {min_count}회 이상)\n파랑: 평균 이상 / 빨강: 평균 미만", fontsize = 12, weight = "bold")
    plt.xlabel("평균 감정지수")
    plt.legend(fontsize = 9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/text_keyword_mood.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("[키워드별 감정지수] 저장 완료")

if __name__ == "__main__":
    set_universal_font()

    parser = argparse.ArgumentParser(description = "텍스트 마이닝 분석 스크립트 (브레인 덤프 컬럼)")
    parser.add_argument(
        "--input",
        type = str,
        default = "notion_brain_dump_raw_cleaned_impute_none.csv",
        help = "입력 CSV 파일 경로 (기본값: notion_brain_dump_raw_cleaned_impute_none.csv)"
    )
    parser.add_argument(
        "--threshold",
        type = float,
        default = 0.3,
        help = "상위/하위 구간 비율 (기본값: 0.3 → 상위 30%% / 하위 30%%)"
    )
    args = parser.parse_args()

    try:
        font_path = get_wordcloud_font()
        df = load_data(args.input)

        stopwords = load_stopwords()

        print("\\n[형태소 분석] 전체 텍스트 토큰 추출 중...")
        all_tokens = extract_tokens(df[TEXT_COL].tolist(), stopwords)
        print(f"  총 {len(all_tokens)}개 토큰 추출 완료")

        # 행별 토큰 (키워드-감정지수 연관 분석용)
        tokens_by_row = {}
        okt = Okt()
        for idx, row in df.iterrows():
            text = row[TEXT_COL]
            if not isinstance(text, str) or not text.strip():
                continue
            morphs = okt.pos(text, norm = True, stem = True)
            tokens_by_row[idx] = [w for w, pos in morphs if pos in TARGET_POS and len(w) > 1 and w not in stopwords]

        # 전체 키워드 빈도
        plot_top_keywords(
            all_tokens,
            title = "전체 기간 상위 키워드 빈도",
            filename = "text_top_keywords.png",
            color = "steelblue"
        )

        # 전체 워드클라우드
        plot_wordcloud(
            all_tokens,
            title = "전체 기간 워드클라우드",
            filename = "text_wordcloud.png",
            font_path = font_path
        )

        # 상위/하위 구간 비교
        plot_high_low_keywords(df, font_path, stopwords, threshold = args.threshold)

        # 키워드별 평균 감정지수
        plot_keyword_mood(df, tokens_by_row)

        print(f"\n[최종 완료] 텍스트 마이닝 시각화가 outputs/ 폴더에 저장되었습니다.")

    except FileNotFoundError as e:
        print(e)
        print("전처리 단계를 먼저 실행하거나 --input 경로를 확인해주세요.")