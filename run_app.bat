@echo off
echo ====================================
echo  4GTP 다중 타겟 예측 시스템
echo ====================================
echo.
echo Streamlit 앱을 시작합니다...
echo.
echo [INFO] 학습된 모델 파일(.pkl)을 업로드하여 사용하세요.
echo [INFO] models/ 폴더에 기존 모델이 있다면 자동으로 로드할 수 있습니다.
echo.
echo 브라우저에서 http://localhost:8501 에 접속하세요.
echo.

cd /d "%~dp0"
streamlit run app/streamlit_app.py --server.port 8501

pause
