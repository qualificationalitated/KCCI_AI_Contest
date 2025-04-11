"""
● 텍스트 입력: 보도자료 또는 정책보고서를 입력받는 간단한 인터페이스 제공
● 자동 요약: AI 기반의 텍스트 자동 요약 서비스 제공 (Transformers 모델 ~ llm 못쓰나? ~ 프롬프트 프로그래밍 연계 하고싶은데)
● 감성 분석: 자료가 갖는 긍정적, 부정적 감정 톤을 자동 판별
● 분야별 영향도 시각화: 산업 분야(예: 노동, 에너지, 금융 등)에 대한 영향을 히트맵 형식으로 시각화 제공
● 시사점 제공: 분석 결과 기반으로 사회적, 산업적 대응 방안과 주요 시사점 제시
● 검수자 의견 시뮬레이션 : 그그, 검수자 의견 내는것도 들어가면 좋을 듯
    검수자 페르소나를 설정해둔 후, 해당 보고자료가 잘 만들어졌는지, 어떤 부분을 추가하면 좋을지 알려주는 기능 있으면 좋겠음
"""

# ✅ KAISIS(KCCI AI Social Impact Simulator)
# KCCI 보도자료 및 정책보고서의 사회적·산업적 영향력 분석 시뮬레이터

import streamlit as st
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, AutoTokenizer
from transformers import pipeline as hf_pipeline
import plotly.express as px
import pandas as pd
import random
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import openai
from openai import OpenAI

# ✅ Streamlit 페이지 설정 (브랜드 적용)
st.set_page_config(
    page_title="KAISIS | 보도자료 영향력 분석 시뮬레이터",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ✅ 페이지 헤더
st.markdown("""
# 📊 KAISIS: KCCI AI Social Impact Simulator
> **대한상공회의소 보도자료·정책보고서의 사회적·산업적 영향력 분석 시뮬레이터**

본 서비스는 생성형 인공지능을 활용하여 KCCI의 보도자료 및 정책 보고서가 우리 사회와 산업에 미치는 정성적·정량적 영향을 분석합니다. 정책 대응을 위한 시사점 도출과 영향도 시각화를 지원합니다.
""")

# ✅ ChatGPT API 설정 (GPT-4o)
openai.api_key = st.secrets.get("openai_api_key", "")
oai_client = OpenAI(api_key=openai.api_key)


# ✅ 모델 로드 및 캐싱
@st.cache_resource
def load_models():
    summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "digit82/kobart-summarization",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
    )
    summarizer_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
    sentiment_analyzer = hf_pipeline("sentiment-analysis")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return summarizer_tokenizer, summarizer_model, sentiment_analyzer, sentiment_tokenizer

tokenizer, model, sentiment_analyzer, sentiment_tokenizer = load_models()

SECTOR_KEYWORDS = {
    # 🎯 경제·산업 전반
    "노동시장": ["고용", "노동", "임금", "일자리", "근로자", "근무", "직장"],
    "금융": ["금리", "대출", "은행", "자본", "투자", "증권", "금융시장", "채권"],
    "부동산": ["건설", "부동산", "공급", "청약", "토지", "재건축", "임대", "매매"],
    "에너지·환경": ["에너지", "탄소", "기후", "온실가스", "재생에너지", "환경", "오염", "지속가능"],
    "무역·통상": ["수출", "수입", "관세", "무역", "통상", "FTA", "WTO", "무역적자"],
    "중소기업·소상공인": ["중소기업", "소상공인", "벤처", "스타트업", "창업", "시장", "상인", "골목경제"],
    "제조업": ["반도체", "자동차", "조선", "기계", "화학", "철강", "산업단지"],
    "서비스업": ["관광", "유통", "물류", "프랜차이즈", "헬스케어", "레저", "호텔", "외식"],
    "디지털·정보통신": ["디지털", "AI", "인공지능", "데이터", "5G", "클라우드", "SW", "ICT", "로봇"],
    "농림수산": ["농업", "축산", "수산", "농민", "어민", "농촌", "식량", "작물", "어획"],
    
    # 👨‍👩‍👧 민생·복지
    "민생·서민생활": ["물가", "생활비", "월세", "전세", "공공요금", "전기세", "교통비", "생계"],
    "복지·건강": ["복지", "건강", "요양", "의료", "건보", "간병", "장애인", "노인", "취약계층"],
    "보건의료": ["병원", "의사", "간호사", "백신", "의약품", "전염병", "코로나", "의료인력"],
    "교육": ["교육", "학교", "대학", "입시", "수능", "청년", "학원", "훈련", "직업교육"],
    "청년·고령": ["청년", "청년층", "청년정책", "고령화", "노인", "시니어", "은퇴", "연금"],

    # 🏛️ 정치·제도
    "규제·행정": ["규제", "완화", "허가", "심사", "인허가", "행정절차", "행정규제", "민원"],
    "정치·선거": ["정당", "국회", "의원", "대통령", "총선", "지방선거", "선거제도"],
    "법률·사법": ["법원", "판결", "검찰", "재판", "형사", "민사", "형량", "헌법", "범죄"],
    "안보·국방": ["군사", "국방", "안보", "북한", "훈련", "전쟁", "전략", "방위비"],

    # 🎨 사회·문화·미디어
    "문화·엔터테인먼트": ["영화", "음악", "공연", "콘텐츠", "드라마", "예능", "K-콘텐츠", "OTT"],
    "언론·미디어": ["언론", "방송", "신문", "유튜브", "SNS", "미디어", "플랫폼"],
    "사회문제": ["양극화", "차별", "혐오", "갈등", "노숙", "폭력", "범죄", "젠더", "이주민"],
    "재난·안전": ["재난", "재해", "지진", "화재", "폭우", "산불", "건축물붕괴", "안전사고"],

    # 🌍 글로벌
    "국제정세": ["외교", "정세", "국제", "유엔", "중국", "미국", "일본", "전쟁", "분쟁", "협상"],
    "해외경제": ["글로벌", "세계경제", "미국경제", "중국경제", "환율", "수요", "공급망"]
}

# ✅ 감성 분석 입력 길이 제한 함수
def truncate_for_sentiment(text, max_length=512):
    inputs = sentiment_tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    return sentiment_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

# ✅ KoBART 기반 요약 함수
def summarize_kobart(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        input_ids,
        max_length=150, min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ 문장에서 산업 키워드별 문장 추출 함수
def extract_sector_sentences(text, keyword_dict):
    sector_sentences = defaultdict(list)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        for sector, keywords in keyword_dict.items():
            if any(keyword in sentence for keyword in keywords):
                sector_sentences[sector].append(sentence)
    return sector_sentences

# ✅ 산업별 감성 분석 실행 함수
def analyze_sector_sentiments(text, sector_keywords, analyzer):
    result = []
    all_sectors = sector_keywords.keys()
    sectors = extract_sector_sentences(text, sector_keywords)
    for sector in all_sectors:
        sentences = sectors.get(sector, [])
        if sentences:
            combined_text = " ".join(sentences)[:512]
            safe_input = truncate_for_sentiment(combined_text)
            sentiment = analyzer(safe_input)[0]
            result.append({
                "분야": sector,
                "감정": sentiment["label"],
                "신뢰도": round(sentiment["score"], 2),
                "문장 수": len(sentences),
                "내용": combined_text
            })
        else:
            result.append({
                "분야": sector,
                "감정": "NEUTRAL",
                "신뢰도": 0.1,
                "문장 수": 0,
                "내용": ""
            })
    return result

# ✅ TF-IDF 기반 영향도 계산 함수
def calculate_sector_tfidf(sector_results):
    corpus = [item["내용"] if item["내용"] else item["분야"] for item in sector_results]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = tfidf_matrix.sum(axis=1).A1
    for i, score in enumerate(scores):
        sector_results[i]["영향도"] = round((score + 0.5) * 10, 2)
    return sector_results

# ✅ 감정 레이블에 따른 색상값 변환 함수
def normalize_score(label, score):
    if label == "POSITIVE":
        return score * 100
    elif label == "NEGATIVE":
        return -score * 100
    else:
        return 0

# ✅ HuggingFace KoAlpaca API를 통한 피드백 생성 함수
def query_koalpaca(prompt):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    try:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "error" in result:
            return f"[⚠️ 오류] 모델 응답 실패: {result['error']}"
        else:
            return "[⚠️ 오류] 예상치 못한 응답 구조입니다."
    except Exception as e:
        return f"[⚠️ 예외 발생] {str(e)}"

# 관리자 피드백 생성 함수 (GPT-4o 호환)
def generate_feedback_with_openai(summary):
    personas = {
        "부회장": "대한상공회의소 부회장으로서, 정책 메시지의 설득력과 공공성 관점에서 평가해 주세요.",
        "전무이사": "대한상공회의소 전무이사로서, 기업 현장의 실행 가능성과 수용성 관점에서 평가해 주세요."
    }
    feedbacks = {}
    for name, persona in personas.items():
        prompt = f"{persona}\n아래는 보도자료 요약입니다.\n\n'{summary}'\n\n이 보도자료에 대해 다음을 작성해 주세요:\n➊ 총평, ➋ 좋은 점, ➌ 부족한 점, ➍ 개선 제안."
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 정책 보고서 검토 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            feedbacks[name] = response.choices[0].message.content.strip()
        except Exception as e:
            feedbacks[name] = f"[⚠️ 오류 발생] {str(e)}"
    return feedbacks

# ✅ 입력란 + 실행 버튼
text_input = st.text_area("✍️ 분석할 보도자료 내용을 입력하세요", height=300)

if st.button("🚀 산업별 영향도 + 감성 분석") and text_input.strip():
    with st.spinner("요약 및 분석 중..."):
        summary = summarize_kobart(text_input)
        sector_results = analyze_sector_sentiments(summary, SECTOR_KEYWORDS, sentiment_analyzer)
        sector_results = calculate_sector_tfidf(sector_results)

        viz_data = {
            "분야": [],
            "영향도": [],
            "색상": []
        }
        for row in sector_results:
            viz_data["분야"].append(row["분야"])
            viz_data["영향도"].append(row["영향도"])
            viz_data["색상"].append(normalize_score(row["감정"], row["신뢰도"]))

    st.subheader("📌 보도자료 요약")
    st.write(summary)

    fig = px.treemap(
        viz_data,
        path=['분야'],
        values='영향도',
        color='색상',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    st.subheader("🌐 산업별 감성 + 영향도 시각화")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 산업별 하위 키워드 목록"):
        for sector, keywords in SECTOR_KEYWORDS.items():
            st.markdown(f"**{sector}**: `{', '.join(keywords)}`")

    st.subheader("🧑‍⚖️ 검수자 피드백")
    with st.spinner("부회장님과 전무이사님의 의견을 생성 중..."):
        feedbacks = generate_feedback_with_openai(summary)
        for name, opinion in feedbacks.items():
            st.markdown(f"### 💬 {name}님의 의견")
            st.info(opinion)