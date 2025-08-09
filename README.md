# GPS_Analysis: GPS 데이터 처리 파이프라인

## 프로젝트 개요

`GPS_Analysis`는 KOTI (한국교통연구원)의 GPS 데이터를 처리하고 분석하기 위한 파이프라인입니다. 이 프로젝트는 원본 GPS 데이터를 파싱하고 필터링하며, 물리 모델 기반의 피처 엔지니어링을 수행하고, 최종적으로 하이브리드 머신러닝 모델을 사용하여 전기차(EV)의 전력 소비를 예측합니다.

## 주요 기능

*   **데이터 파싱 및 필터링**: 원본 CSV 형식의 GPS 데이터를 읽어 유효한 데이터를 추출하고 필터링합니다.
*   **피처 엔지니어링**: GPS 데이터로부터 다양한 특징(예: 속도, 가속도, 고도 변화)을 추출하고, 기상청(KMA) API를 활용하여 기상 데이터를 연동합니다.
*   **물리 모델 기반 전력 계산**: 차량 모델별 파라미터를 기반으로 물리적인 전력 소비량을 계산합니다.
*   **하이브리드 모델 예측**: 머신러닝 모델을 사용하여 전력 소비를 예측합니다.
*   **CLI 기반 인터페이스**: 각 처리 단계를 개별적으로 실행하거나 전체 파이프라인을 순차적으로 실행할 수 있는 사용자 친화적인 명령줄 인터페이스를 제공합니다.

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/ws-b/GPS_Analysis.git
cd GPS_Analysis
```

### 2. Python 환경 설정

Python 3.8 이상을 권장합니다. 가상 환경을 사용하는 것을 강력히 추천합니다.

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치

`requirements.txt`에 명시된 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 기상청(KMA) API 키를 추가합니다.

```
KMA_API_KEY="YOUR_KMA_API_KEY_HERE"
```

**참고**: 기상청 API 키는 [기상청 날씨누리](https://data.kma.go.kr/)에서 발급받을 수 있습니다.

### 5. 데이터 준비

*   **GPS 데이터**
*   **기상 관측소 데이터**: `Source/stations.csv` 파일
*   **ML 모델 및 스케일러**: `Source/config.py`에 정의된 `model_path` 및 `scaler_path` 경로에 학습된 머신러닝 모델 파일(`.model`)과 스케일러(`.pkl`) 파일이 있어야 합니다.

## 사용법

프로젝트의 메인 스크립트 `main.py`를 실행하여 대화형 메뉴를 통해 파이프라인을 실행할 수 있습니다.

```bash
python main.py
```

## 프로젝트 구조

```
GPS_Analysis/
├── .env                  # 환경 변수 (KMA_API_KEY 등)
├── .gitignore            # Git 무시 파일
├── main.py               # 메인 실행 스크립트 (CLI 인터페이스)
├── requirements.txt      # Python 의존성 목록
└── Source/               # 핵심 로직 및 설정 파일
    ├── config.py         # 경로, API 키, 차량 파라미터 등 설정
    ├── data_handler.py   # 데이터 파싱, 피처 엔지니어링, 전력 계산 로직
    └── stations.csv      # 기상 관측소 정보 (위도, 경도)
```