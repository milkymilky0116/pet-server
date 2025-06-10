
🐶 애견 추천 시스템 API - 설계 및 목적 설명 (서버 IT 담당자용)

이 시스템은 FastAPI를 기반으로 구성된 애완견 추천 엔진입니다.
총 1200마리 강아지에 대한 특성과 선호 정보를 반영하여, 사용자 입력에 따라 가장 잘 맞는 강아지를 추천합니다.

===============================
📌 주요 설계 목적 및 구조 설명
===============================

1. 목적:
- 사용자(예비 반려인)의 애완견 특성 + 성향 + 위치 정보를 입력받아
- 1200개 강아지 중에서 가장 적합한 강아지 10마리를 추천
- 강아지의 선호 정보(preference)도 반영한 양방향 추천 모델을 학습 및 적용
- 관리자용 평가 API도 제공하여 모델 정합도 및 개선사항 확인 가능

2. 사용 기술:
- Python 3 + FastAPI
- PyTorch 기반 딥러닝 추천 모델 (RankNet 구조)
- Pandas, scikit-learn, Numpy 등 전처리 및 분석 라이브러리

3. API 구성:
- GET    /                     → 서버 상태 확인
- POST   /recommend           → 추천 결과 (10마리 + GPS + 이미지)
- GET    /admin/evaluate      → 모델 성능 평가 지표 반환

4. 추천 모델 구조:
- models/ranknet_bidirectional.py → 양방향 추천 모델 (강아지 preference 반영)
- models/bi_ranknet.pt            → 학습된 딥러닝 모델 파라미터
- 모델은 실시간 예측 시 이 파라미터를 불러와 점수 예측

5. 사용자 입력 구조 (예시):
{
  "name": "초코",
  "age_month": 36,
  "weight": 5.2,
  "color": "Beige",
  "personality": "사교적",
  "region": "서울",
  "vaccinated": "코로나, 광견병",
  "preference_age_range": "20-60",
  "preference_weight_range": "4.0-7.0",
  "preference_color": "Cream",
  "preference_personality": "온순함",
  "preference_region": "수도권",
  "preference_vaccine": "코로나, 광견병",
  "lat": 37.55,
  "lon": 126.97
}

6. 동작 방식:
- 입력을 벡터로 변환 (utils/preprocess.py)
- 모델을 통해 1200마리 전부에 대해 추천 점수 예측
- 상위 10마리를 점수 기준으로 추출하여 추천 결과 구성
- 결과에는 이미지 파일명(ID.png), GPS 위도/경도 포함

7. 평가 기능:
- utils/evaluator.py → 내부적으로 permutation importance 등 평가 가능
- GET /admin/evaluate → 실시간으로 모델 평가 확인 가능

========================
📁 압축 파일 구성 요약
========================

- main.py                  → FastAPI 엔트리 포인트
- models/
    ├── ranknet_unidirectional.py
    ├── ranknet_bidirectional.py  ✅ .pt 파일 로딩 포함
    └── bi_ranknet.pt             ✅ 학습된 모델
- utils/
    ├── preprocess.py
    └── evaluator.py
- data/
    ├── dog_dummy_data_1200_with_gps_final.xlsx
    └── images/                   (ID.png 파일 저장 위치)
- README_애견추천API구성요약.txt  ← 전체 구성 요약 파일

========================
🧠 IT 담당자 주의사항
========================

- 서버 실행 전에 bi_ranknet.pt가 존재해야 합니다.
- 향후 모델 개선 시 .pt 파일만 교체하면 서버는 유지됩니다.
- 이미지 폴더(data/images/)에는 각 강아지의 ID.png 파일이 있어야 UI 연동에 문제가 없습니다.

========================
🔗 실행 방법
========================
$ cd recommendation_server
$ uvicorn main:app --reload

→ http://127.0.0.1:8000/docs 에서 Swagger 문서 확인 가능
