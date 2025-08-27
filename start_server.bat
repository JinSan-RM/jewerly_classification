# Windows용 실행 스크립트 (start_server.bat)
@echo off
echo 🚀 주얼리 AI 분류 시스템 시작 중...
echo.

REM 가상환경 활성화 확인
if not exist "jewelry_classifier\Scripts\activate.bat" (
    echo 가상환경을 찾을 수 없습니다.
    echo 먼저 다음 명령어로 가상환경을 생성하세요:
    echo python -m venv jewelry_classifier
    echo jewelry_classifier\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM 가상환경 활성화
call jewelry_classifier\Scripts\activate.bat

REM 모델 파일 확인
if not exist "jewerly_classification_AI_model.pth" (
    echo 모델 파일을 찾을 수 없습니다: jewerly_classification_AI_model.pth
    echo 훈련된 모델 파일을 현재 폴더에 복사해주세요.
    pause
    exit /b 1
)

echo 환경 설정 완료
echo 브라우저에서 http://localhost:8000 으로 접속하세요
echo.

REM Flask 서버 시작
python app.py

pause

