@echo off
REM Bookscan 실행. 처음 실행하면 가상환경과 의존성을 자동으로 준비한다.
REM 주의: 이 파일은 CRLF 줄바꿈 + CP949(euc-kr) 인코딩으로 저장해야 한다.
REM   - LF만 있으면 cmd가 파싱에 실패한다.
REM   - UTF-8로 저장하면 한글이 깨지고, chcp 65001을 파일 안에서 걸면
REM     cmd가 다음 줄의 바이트 위치를 잘못 잡아 엉뚱한 명령을 실행한다.
REM   .gitattributes에서 CRLF와 binary(변환 금지)를 강제하고 있다.
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" goto :setup
REM 가상환경은 있는데 설치가 중간에 끊긴 경우를 잡는다.
"%PY%" -c "import cv2, fastapi, img2pdf, PIL" >nul 2>nul
if not errorlevel 1 goto :run
echo [Bookscan] 의존성이 빠져 있습니다. 다시 설치합니다.
goto :deps

:setup
echo [Bookscan] 가상환경을 만드는 중...
set "UVPY=%APPDATA%\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe"
if exist "%UVPY%" "%UVPY%" -m venv .venv
if exist "%PY%" goto :deps

where py >nul 2>nul
if not errorlevel 1 py -3 -m venv .venv
if exist "%PY%" goto :deps

python -m venv .venv 2>nul
if exist "%PY%" goto :deps
goto :nopython

:deps
echo [Bookscan] 의존성을 설치하는 중... ^(처음 한 번만, 몇 분 걸립니다^)
"%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto :depsfail

:run
echo [Bookscan] 서버를 시작합니다. 브라우저가 자동으로 열립니다.
echo [Bookscan] 종료하려면 이 창에서 Ctrl+C 를 누르세요.
"%PY%" -m bookscan %*
goto :end

:nopython
echo.
echo [Bookscan] 파이썬을 찾을 수 없습니다.
echo            Python 3.10 이상을 설치한 뒤 다시 실행하세요.
echo            https://www.python.org/downloads/
echo            설치할 때 Add python.exe to PATH 를 반드시 체크하세요.
pause
exit /b 1

:depsfail
echo.
echo [Bookscan] 의존성 설치에 실패했습니다.
echo            인터넷 연결을 확인한 뒤 다시 실행하세요.
pause
exit /b 1

:end
endlocal
