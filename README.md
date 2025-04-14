✅ README.md 제안

# 📊 KAISIS: KCCI AI Social Impact Simulator

대한상공회의소(KCCI)가 발간하는 보도자료 및 정책 보고서의 사회적·산업적 영향을  
생성형 인공지능 기반으로 실시간 분석하고 시각화하는 Streamlit 기반 웹 애플리케이션입니다.

---

## 🧭 서비스 개요

> "KAISIS는 보도자료가 대한민국 사회에 미치는 영향력을 정량·정성 분석하여  
> 정책 대응과 커뮤니케이션을 한눈에 정리할 수 있는 지능형 시뮬레이터입니다."

- **자동 요약**: GPT-4o를 활용한 핵심 요약 제공  
- **감성 분석**: 산업별 긍/부정/중립 감성 분석  
- **분야 분류**: 보도자료의 문맥에 따라 자동 섹터 태깅  
- **영향도 시각화**: Treemap을 통한 강도·방향성 시각화  
- **검수자 피드백**: 가상의 부회장·전무이사 페르소나 피드백 생성

---

## 🛠️ 기술 스택

| 항목         | 기술 |
|--------------|------|
| 프론트엔드   | [Streamlit](https://streamlit.io)  
| 백엔드 모델  | HuggingFace Transformers (감성 분석), GPT-4o (요약/피드백)  
| 시각화       | Plotly Treemap  
| NLP 기능     | KoBART 요약, TF-IDF 기반 영향도 계산  
| 배포 환경    | Streamlit Cloud

---

## 🚀 실행 방법

1. 리포지토리 클론

```bash
git clone https://github.com/your-org/kcciaisis.git
cd kcciaisis
```
2. 의존성 설치

```bash
pip install -r requirements.txt
```

3. API 키 등록 (.streamlit/secrets.toml)
```toml
openai_api_key = "sk-..."
```

4. 실행
```bash
streamlit run impact_simulator.py
```
👥 기획·개발
KCCI 대한상공회의소 IT지원팀 김현우
