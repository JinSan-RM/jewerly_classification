# Linux/Mac용 실행 스크립트 (start_server.sh)
#!/bin/bash

echo "주얼리 AI 분류 시스템 시작 중"
echo

# 가상환경 확인
if [ ! -f "jewelry_classifier/bin/activate" ]; then
    echo "가상환경을 찾을 수 없습니다."
    echo "먼저 다음 명령어로 가상환경을 생성하세요:"
    echo "python -m venv jewelry_classifier"
    echo "source jewelry_classifier/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# 가상환경 활성화
source jewelry_classifier/bin/activate

# 모델 파일 확인
if [ ! -f "jewerly_classification_AI_model.pth" ]; then
    echo "모델 파일을 찾을 수 없습니다: jewerly_classification_AI_model.pth"
    echo "훈련된 모델 파일을 현재 폴더에 복사해주세요."
    exit 1
fi

echo "✅ 환경 설정 완료"
echo "📂 브라우저에서 http://localhost:8000 으로 접속하세요"
echo

# Flask 서버 시작
python app.py

---

# 패키지 설치 스크립트 (install_packages.bat for Windows)
@echo off
echo 📦 패키지 설치 중...

REM 가상환경 생성
echo 가상환경 생성 중...
python -m venv jewelry_classifier

REM 가상환경 활성화
call jewelry_classifier\Scripts\activate.bat

REM 패키지 설치
echo 필요한 패키지들을 설치하고 있습니다...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ 설치 완료!
echo 이제 start_server.bat을 실행하여 서버를 시작할 수 있습니다.
pause

---

# 패키지 설치 스크립트 (install_packages.sh for Linux/Mac)
#!/bin/bash

echo "패키지 설치 중"

# 가상환경 생성
echo "가상환경 생성 중"
python -m venv jewelry_classifier

# 가상환경 활성화
source jewelry_classifier/bin/activate

# 패키지 설치
echo "필요한 패키지들을 설치하고 있습니다."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "설치 완료"
echo "이제 start_server.sh를 실행하여 서버를 시작할 수 있습니다."
echo "chmod +x start_server.sh"
echo "./start_server.sh"