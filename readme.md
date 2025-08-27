# 주얼리 AI 자동 분류 시스템

이 프로젝트는 ConvNeXt 기반의 딥러닝 모델을 사용하여 주얼리 이미지를 자동으로 분류하는 웹 애플리케이션입니다.

## 요구사항

- Python 3.8+
- PyTorch 2.0+
- 훈련된 모델 파일: `memory_optimized_best_model.pth`

## 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv jewelry_classifier

# 가상환경 활성화 (Windows)
soruce jewelry_classifier\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
source jewelry_classifier/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 프로젝트 구조 설정

프로젝트 폴더를 다음과 같이 구성하세요:

```
jewelry_classifier/
├── app.py                              # Flask 백엔드 서버
├── main.html                           # 웹 인터페이스
├── jewerly_classification_AI_model.pth     # 훈련된 모델 파일
├── requirements.txt                    # 패키지 의존성
├── uploads/                            # 업로드된 이미지 저장 폴더 (자동 생성)
└── results/                            # 분석 결과 저장 폴더 (자동 생성)
```

### 4. 모델 파일 확인

- `jewerly_classification_AI_model.pth` 파일이 프로젝트 루트 디렉토리에 있는지 확인하세요.
- 파일이 없다면 훈련된 모델 파일을 해당 위치에 복사하세요.

### 5. 서버 실행

```bash
python app.py
```

서버가 정상적으로 시작되면 다음과 같은 메시지가 출력됩니다:

```
주얼리 분류 서버 시작 중...
사용 장치: cuda  # 또는 cpu
ConvNeXt 백본 로드: convnext_base.fb_in22k_ft_in1k
   백본 특성 차원: 1024
   state_dict 로드 성공 (strict=True)
   모델 로드 및 설정 완료!
모델 초기화 완료
서버 준비 완료!
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://[your-ip]:5000
```

### 6. 웹 애플리케이션 접속

브라우저에서 `http://localhost:5000`으로 접속하세요.

## 사용 방법

1. **이미지 업로드**: 주얼리 이미지를 드래그하거나 파일 선택 버튼을 클릭하여 업로드
2. **AI 분류**: "AI 분류 시작" 버튼을 클릭하여 이미지 분석 시작
3. **결과 확인**