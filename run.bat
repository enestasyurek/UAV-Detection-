@echo off
echo Drone Tespit ve Takip Sistemi Baslatiliyor...
echo.

REM Python kontrolu
python --version >nul 2>&1
if errorlevel 1 (
    echo HATA: Python yuklu degil veya PATH'e eklenmemis!
    echo Lutfen Python 3.8 veya uzerini yukleyin.
    pause
    exit /b 1
)

REM Sanal ortam kontrolu
if not exist "venv" (
    echo Sanal ortam olusturuluyor...
    python -m venv venv
    echo.
)

REM Sanal ortami aktif et
echo Sanal ortam aktif ediliyor...
call venv\Scripts\activate.bat

REM Bagimliliklari yukle
echo.
echo Bagimliliklar kontrol ediliyor...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo Bagimliliklar yukleniyor...
    pip install -r requirements.txt
)

REM Uygulamayi baslat
echo.
echo Uygulama baslatiliyor...
echo.
python src/main.py %*

REM Hata kontrolu
if errorlevel 1 (
    echo.
    echo HATA: Uygulama calistirilirken hata olustu!
    pause
)