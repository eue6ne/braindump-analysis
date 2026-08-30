import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_visualization import set_universal_font
from analysis.analysis_text import (
    load_stopwords,
    extract_tokens,
    TARGET_POS
)
from wordcloud import WordCloud

st.set_page_config(page_title = "텍스트 마이닝", page_icon = "📝", layout = "wide")

st.title("📝 텍스트 마이닝")
st.markdown("---")

TEXT_COL = "브레인 덤프"
TARGET_COL = "감정지수"

if st.session_state.get("cleaned_df") is None:
    st.warning("먼저 데이터 수집 및 전처리를 완료해주세요.")
    st.stop()

df = st.session_state["cleaned_df"].copy()
set_universal_font()

if TEXT_COL not in df.columns:
    st.error(f"'{TEXT_COL}' 컬럼이 없습니다.")
    st.stop()

st.success(f"✅ 전처리된 데이터 로드 완료 — {len(df)}행")

def get_font_path():
    import platform
    candidates = {
        "Darwin": ["/Library/Fonts/AppleGothic.ttf",
                   "/System/Library/Fonts/Supplemental/AppleGothic.ttf"],
        "Windows": ["C:/Windows/Fonts/malgun.ttf"],
    }
    for path in candidates.get(platform.system(), []):
        if os.path.exists(path):
            return path
    for path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if os.path.exists(path):
            return path
    return None

col1, col2 = st.columns(2)
with col1:
    threshold = st.slider("상위/하위 구간 비율", min_value = 0.1, max_value = 0.5, value = 0.3, step = 0.05,
                           help = "감정지수 상위/하위 N%를 비교합니다.")
with col2:
    stopwords_path = st.text_input("불용어 파일 경로", value = "analysis/stopwords.txt")

if st.button("📝 텍스트 마이닝 실행", use_container_width = True, type = "primary"):
    from konlpy.tag import Okt

    font_path = get_font_path()
    if not font_path:
        st.error("한글 워드클라우드 폰트를 찾을 수 없습니다.")
        st.stop()

    stopwords = load_stopwords(stopwords_path)
    df_text = df.dropna(subset = [TEXT_COL])

    with st.spinner("형태소 분석 중... (시간이 걸릴 수 있습니다)"):
        okt = Okt()
        all_tokens = extract_tokens(df_text[TEXT_COL].tolist(), stopwords)

        tokens_by_row = {}
        for idx, row in df_text.iterrows():
            text = row[TEXT_COL]
            if not isinstance(text, str) or not text.strip():
                continue
            morphs = okt.pos(text, norm = True, stem = True)
            tokens_by_row[idx] = [w for w, pos in morphs if pos in TARGET_POS and len(w) > 1 and w not in stopwords]

    st.success(f"✅ 총 {len(all_tokens)}개 토큰 추출 완료")

    # 전체 키워드 빈도
    st.markdown("---")
    st.markdown("## 🔤 전체 기간 상위 키워드 빈도")
    counter = Counter(all_tokens)
    top = counter.most_common(20)
    if top:
        words, counts = zip(*top)
        fig, ax = plt.subplots(figsize = (10, 8))
        ax.barh(list(reversed(words)), list(reversed(counts)), color = "steelblue", alpha = 0.8, edgecolor = "white")
        ax.set_title("전체 기간 상위 키워드 빈도", fontsize = 12, weight = "bold")
        ax.set_xlabel("빈도")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 워드클라우드
    st.markdown("## ☁️ 전체 기간 워드클라우드")
    wc = WordCloud(font_path = font_path, background_color = "white",
                   width = 800, height = 500, max_words = 100, colormap = "coolwarm").generate_from_frequencies(counter)
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.imshow(wc, interpolation = "bilinear")
    ax.axis("off")
    ax.set_title("전체 기간 워드클라우드", fontsize = 13, weight = "bold", pad = 15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 상위/하위 구간 비교
    st.markdown(f"## 🔵🔴 감정지수 상위 vs 하위 {int(threshold*100)}% 키워드 비교")
    high_q = df_text[TARGET_COL].quantile(1 - threshold)
    low_q = df_text[TARGET_COL].quantile(threshold)
    high_tokens = extract_tokens(df_text[df_text[TARGET_COL] >= high_q][TEXT_COL].tolist(), stopwords)
    low_tokens = extract_tokens(df_text[df_text[TARGET_COL] <= low_q][TEXT_COL].tolist(), stopwords)
    high_counter = Counter(high_tokens)
    low_counter = Counter(low_tokens)

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    for ax, c, title, cmap in zip(axes, [high_counter, low_counter],
                                   [f"상위 {int(threshold*100)}% 키워드", f"하위 {int(threshold*100)}% 키워드"],
                                   ["Blues", "Reds"]):
        if c:
            wc = WordCloud(font_path = font_path, background_color = "white",
                           width = 500, height = 350, max_words = 60, colormap = cmap).generate_from_frequencies(c)
            ax.imshow(wc, interpolation = "bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize = 12, weight = "bold")
    plt.suptitle("감정지수 상위 vs 하위 구간 키워드 비교", fontsize = 13, weight = "bold", y = 1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 키워드 차이
    all_words = set(high_counter.keys()) | set(low_counter.keys())
    diff_data = [{"단어": w, "차이": high_counter.get(w, 0) - low_counter.get(w, 0)}
                 for w in all_words if high_counter.get(w, 0) + low_counter.get(w, 0) >= 3]
    if diff_data:
        diff_df = pd.DataFrame(diff_data).sort_values("차이", ascending = False)
        top_high_w = diff_df.head(15)
        top_low_w = diff_df.tail(15).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize = (14, 6))
        axes[0].barh(top_high_w["단어"], top_high_w["차이"], color = "steelblue", alpha = 0.8, edgecolor = "white")
        axes[0].set_title("감정 높은 날 더 많이 등장한 단어", fontsize = 11, weight = "bold")
        axes[0].axvline(x = 0, color = "black", linewidth = 0.8, linestyle = "--")
        axes[1].barh(top_low_w["단어"], top_low_w["차이"].abs(), color = "crimson", alpha = 0.8, edgecolor = "white")
        axes[1].set_title("감정 낮은 날 더 많이 등장한 단어", fontsize = 11, weight = "bold")
        plt.suptitle("감정지수 상위 vs 하위 구간 키워드 차이 분석", fontsize = 13, weight = "bold", y = 1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 키워드별 평균 감정지수
    st.markdown("## 😊 키워드별 평균 감정지수")
    word_mood = {}
    for idx, tokens in tokens_by_row.items():
        mood = df_text.loc[idx, TARGET_COL]
        for word in set(tokens):
            word_mood.setdefault(word, []).append(mood)
    mood_data = [{"단어": w, "평균감정지수": np.mean(moods), "등장횟수": len(moods)}
                 for w, moods in word_mood.items() if len(moods) >= 3]
    if mood_data:
        mood_df = pd.DataFrame(mood_data).sort_values("평균감정지수", ascending = False)
        top_df = pd.concat([mood_df.head(10), mood_df.tail(10)]).drop_duplicates()
        overall_mean = df_text[TARGET_COL].mean()
        bar_colors = ["steelblue" if v >= overall_mean else "crimson" for v in top_df["평균감정지수"]]
        fig, ax = plt.subplots(figsize = (9, max(6, len(top_df) * 0.45)))
        ax.barh(top_df["단어"], top_df["평균감정지수"], color = bar_colors, alpha = 0.8, edgecolor = "white")
        ax.axvline(x = overall_mean, color = "black", linestyle = "--", linewidth = 1.2,
                   label = f"전체 평균 ({overall_mean:.1f})")
        ax.set_title(f"키워드별 평균 감정지수 (등장 3회 이상)\n파랑: 평균 이상 / 빨강: 평균 미만",
                     fontsize = 12, weight = "bold")
        ax.set_xlabel("평균 감정지수")
        ax.legend(fontsize = 9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # NLP 감정 분석
    st.markdown("## 🤖 NLP 감정 분석 (보조 분석)")
    lexicon_path = "data/KnuSentiLex/data/SentiWord_info.json"
    if not os.path.exists(lexicon_path):
        st.warning(f"KNU 감성사전 파일을 찾을 수 없습니다: `{lexicon_path}`\n\n`git clone https://github.com/park1200656/KnuSentiLex.git data/KnuSentiLex` 를 실행하세요.")
    else:
        with open(lexicon_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
        lexicon = {}
        for entry in data:
            word = entry.get("word_root", "").strip()
            score = int(entry.get("polarity", 0))
            if word and (word not in lexicon or abs(score) > abs(lexicon[word])):
                lexicon[word] = score

        import matplotlib.dates as mdates
        df_text = df_text.copy()
        df_text["감정점수_NLP"] = df_text[TEXT_COL].apply(
            lambda t: float(np.mean([lexicon[w] for w in okt.morphs(t, norm=True, stem=True) if w in lexicon]))
            if isinstance(t, str) and any(w in lexicon for w in okt.morphs(t, norm=True, stem=True)) else 0.0
        )

        corr = df_text["감정점수_NLP"].corr(df_text[TARGET_COL])
        col1, col2 = st.columns(2)
        col1.metric("NLP 감정 점수 평균", f"{df_text['감정점수_NLP'].mean():.3f}")
        col2.metric("두 지표 상관계수", f"{corr:.3f}")

        ma7 = df_text["감정점수_NLP"].rolling(window = 7, center = True).mean()
        fig, ax = plt.subplots(figsize = (12, 5))
        ax.fill_between(df_text.index, df_text["감정점수_NLP"], 0,
                        where = df_text["감정점수_NLP"] >= 0, color = "crimson", alpha = 0.3, label = "긍정 구간")
        ax.fill_between(df_text.index, df_text["감정점수_NLP"], 0,
                        where = df_text["감정점수_NLP"] < 0, color = "steelblue", alpha = 0.3, label = "부정 구간")
        ax.plot(df_text.index, df_text["감정점수_NLP"], color = "gray", linewidth = 0.8, alpha = 0.6)
        ax.plot(ma7.index, ma7.values, color = "black", linewidth = 2.0, label = "7일 이동평균")
        ax.axhline(y = 0, color = "black", linestyle = "--", linewidth = 0.8)
        ax.set_title("일별 NLP 감정 점수 추이", fontsize = 12, weight = "bold")
        ax.set_ylabel("감정 점수 (-2 ~ 2)")
        ax.legend(fontsize = 9)
        ax.grid(True, linestyle = ":", alpha = 0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation = 35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✅ 텍스트 마이닝 완료!")