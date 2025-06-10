
애견 추천 API 구성 요약 (모델 포함 최종버전)

📁 전체 구조:
├── main.py                        # FastAPI 서버 진입점
├── models/
│   ├── ranknet_unidirectional.py # 일방향 추천 모델 정의
│   ├── ranknet_bidirectional.py  # 양방향 추천 모델 정의 + bi_ranknet.pt 로드
│   └── bi_ranknet.pt             # 학습된 RankNet 양방향 모델 파라미터
├── utils/
│   ├── preprocess.py             # 사용자 + 강아지 벡터 전처리
│   └── evaluator.py              # 추천모델 평가 지표 계산
├── data/
│   ├── dog_dummy_data_1200_with_gps_final.xlsx  # 강아지 1200마리 더미 데이터
│   └── images/                   # 강아지 이미지 (ID.png 형태) 저장 폴더
└── README.txt (현재 파일)         # 구성 요약 설명

📌 API 요약:
- GET    /                     → 서버 상태 확인
- POST   /recommend           → 사용자 입력 기반 추천 (Top 10 마리 + 이미지/GPS)
- GET    /admin/evaluate      → 일방향/양방향 모델 성능 평가 결과 반환

📌 실행 방법 (로컬 기준):
$ cd recommendation_server
$ uvicorn main:app --reload
→ http://127.0.0.1:8000/docs 에서 Swagger UI 확인 가능
