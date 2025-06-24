@echo off
title Drone Detection System - Windows Launcher
echo ========================================
echo Drone Detection System - Windows Launcher
echo ========================================

REM Python sürümünü kontrol et
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python bulunamadı. Python 3.8+ yüklü olduğundan emin olun.
    echo Python indirmek için: https://python.org/downloads
    pause
    exit /b 1
)

REM Python sürüm kontrolü
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python sürümü: %PYTHON_VERSION%

REM Minimum sürüm kontrolü (Python 3.8+)
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.8+ gerekli. Lütfen Python'ı güncelleyin.
    pause
    exit /b 1
)
echo [OK] Python sürümü uygun

REM Virtual environment kontrolü
if not exist "venv" (
    echo Virtual environment oluşturuluyor...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Virtual environment oluşturulamadı.
        pause
        exit /b 1
    )
)

REM Virtual environment'ı aktifleştir
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Virtual environment aktifleştirilemedi.
    pause
    exit /b 1
)

REM Pip güncellemesi
echo Pip güncelleniyor...
python -m pip install --upgrade pip --quiet

REM Windows için özel gereksinimler
if not exist ".windows_deps_installed" (
    echo Windows bağımlılıkları kuruluyor...
    
    REM OpenCV ve temel kütüphaneler
    pip install opencv-python pillow numpy --quiet
    
    REM Visual C++ Runtime kontrolü
    python -c "import cv2" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] OpenCV yüklenemedi. Visual C++ Redistributable gerekli olabilir.
        echo İndirmek için: https://aka.ms/vs/17/release/vc_redist.x64.exe
    )
    
    REM DPI Awareness test
    python -c "import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] DPI Awareness ayarlanamadı. Windows 8.1+ gerekli.
    )
    
    REM İşaret dosyası oluştur
    echo. > .windows_deps_installed
)

REM Gereksinimler dosyasını kur
if exist "requirements.txt" (
    echo Bağımlılıklar kuruluyor...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Bağımlılıklar kurulamadı.
        pause
        exit /b 1
    )
) else (
    echo requirements.txt bulunamadı. Temel bağımlılıklar kuruluyor...
    pip install opencv-python pillow numpy ultralytics torch torchvision --quiet
    if errorlevel 1 (
        echo [ERROR] Temel bağımlılıklar kurulamadı.
        pause
        exit /b 1
    )
)

REM Python path ayarla
set PYTHONPATH=%PYTHONPATH%;%CD%\src

REM Windows için özel çevre değişkenleri
set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
set PYTHONIOENCODING=utf-8

echo ========================================
echo Uygulama başlatılıyor...
echo Windows optimizasyonları aktif
echo ========================================

REM Uygulamayı başlat
cd src
python main.py
set EXIT_CODE=%ERRORLEVEL%

REM Geri dön
cd ..

REM Virtual environment'tan çık
call venv\Scripts\deactivate.bat

REM Sonuç
if %EXIT_CODE% equ 0 (
    echo ========================================
    echo Uygulama başarıyla kapatıldı.
    echo ========================================
) else (
    echo ========================================
    echo Uygulama hata ile kapatıldı (Kod: %EXIT_CODE%)
    echo ========================================
    echo.
    echo Sorun giderme:
    echo 1. Python 3.8+ yüklü olduğundan emin olun
    echo 2. Visual C++ Redistributable yükleyin
    echo 3. Windows Defender'ı kontrol edin
    echo 4. Yönetici olarak çalıştırmayı deneyin
    echo.
    pause
)

exit /b %EXIT_CODE%