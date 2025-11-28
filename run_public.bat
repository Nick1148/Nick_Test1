@echo off
echo ====================================
echo  4GTP 다중 타겟 예측 시스템
echo  외부 접속 가능 배포 모드
echo ====================================
echo.
echo [INFO] ngrok을 사용하여 외부 URL을 생성합니다.
echo [INFO] 인터넷이 연결되어 있어야 합니다.
echo.

cd /d "%~dp0"
python run_public.py

pause
