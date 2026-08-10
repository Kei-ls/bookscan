@echo off
REM Bookscan 실행. 처음 실행하면 가상환경과 의존성을 자동으로 준비한다.
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [Bookscan] 가상환경을 만드는 중...
    set BASEPY=%APPDATA%\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe
    if not exist "!BASEPY!" set BASEPY=python
    "%APPDATA%\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe" -m venv .venv 2>nul || python -m venv .venv
    if not exist "%PY%" (
        echo [Bookscan] 파이썬을 찾을 수 없습니다. Python 3.10 이상을 설치한 뒤 다시 실행하세요.
        pause
        exit /b 1
    )
    echo [Bookscan] 의존성을 설치하는 중... ^(처음 한 번만, 몇 분 걸립니다^)
    "%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
)

echo [Bookscan] 서버를 시작합니다. 브라우저가 자동으로 열립니다.
echo [Bookscan] 종료하려면 이 창에서 Ctrl+C 를 누르세요.
"%PY%" -m bookscan
endlocal
