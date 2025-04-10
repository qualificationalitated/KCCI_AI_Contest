
import streamlit as st
from transformers import pipeline
import plotly.express as px
import random

st.set_page_config(page_title="KCCI 보도자료 영향력 시뮬레이터", layout="wide")

st.title("📊 KCCI 보도자료 영향력 시뮬레이터")
st.markdown("보도자료가 한국 산업과 사회에 미치는 영향을 자동 분석하고 시각화합니다.")

# 입력
text_input = st.text_area("✍️ 분석할 보도자료 내용을 입력하세요", height=300)

if text_input:
    with st.spinner("요약 및 감성 분석 중..."):
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentiment = sentiment_analyzer(summary)[0]

    st.subheader("📝 요약 결과")
    st.success(summary)

    st.subheader("📈 감성 분석")
    st.info(f"감정: **{sentiment['label']}** (신뢰도: {sentiment['score']:.2f})")

    # 영향도 시뮬레이션 (랜덤 기반 예시)
    st.subheader("🔍 예상 영향 분야 및 강도")
    sectors = ["중소기업", "무역", "노동시장", "에너지", "규제개혁", "교육", "디지털", "환경", "부동산", "금융"]
    impact_data = {
        "분야": sectors,
        "긍정 영향": [random.randint(0, 100) for _ in sectors],
        "부정 영향": [random.randint(0, 100) for _ in sectors]
    }

    fig = px.treemap(
        impact_data,
        path=['분야'],
        values=[max(p, n) for p, n in zip(impact_data["긍정 영향"], impact_data["부정 영향"])],
        color=[p - n for p, n in zip(impact_data["긍정 영향"], impact_data["부정 영향"])],
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 시사점")
    st.markdown(f"이 보도자료는 **{sentiment['label']}**의 정서를 띠며, 특히 `{sectors[random.randint(0,9)]}` 분야에 높은 영향을 줄 것으로 보입니다.")
