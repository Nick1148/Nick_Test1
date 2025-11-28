"""
외부 접속 가능한 배포 스크립트
================================
ngrok을 사용하여 외부에서 접속 가능한 URL을 생성합니다.
"""

import subprocess
import sys
import time
import threading
import webbrowser

def run_streamlit():
    """Streamlit 앱 실행"""
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--server.headless", "true"
    ])

def run_ngrok():
    """ngrok 터널 실행"""
    try:
        from pyngrok import ngrok

        # 기존 터널 종료
        ngrok.kill()

        print("\n" + "="*60)
        print("  4GTP 다중 타겟 예측 시스템 - 외부 배포")
        print("="*60)

        # 잠시 대기 (Streamlit 시작 대기)
        print("\n[INFO] Streamlit 서버 시작 대기 중...")
        time.sleep(5)

        # ngrok 터널 생성
        print("[INFO] ngrok 터널 생성 중...")
        public_url = ngrok.connect(8501)

        print("\n" + "="*60)
        print("  접속 URL")
        print("="*60)
        print(f"\n  [PUBLIC] 외부 접속 URL: {public_url}")
        print(f"  [LOCAL]  로컬 접속 URL: http://localhost:8501")
        print(f"  [LAN]    내부망 URL:    http://192.168.45.214:8501")
        print("\n" + "="*60)
        print("  위 URL을 다른 사람에게 공유하세요!")
        print("  종료하려면 Ctrl+C를 누르세요.")
        print("="*60 + "\n")

        # 브라우저에서 열기
        webbrowser.open(str(public_url))

        # 계속 실행
        while True:
            time.sleep(1)

    except ImportError:
        print("[ERROR] pyngrok이 설치되어 있지 않습니다.")
        print("        pip install pyngrok 로 설치하세요.")
    except Exception as e:
        print(f"[ERROR] ngrok 오류: {e}")

if __name__ == "__main__":
    print("\n[START] 배포 시작...")

    # Streamlit을 별도 스레드에서 실행
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()

    # ngrok 실행
    run_ngrok()
