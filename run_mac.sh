#!/bin/bash

# macOS için Drone Detection System başlatma scripti
echo "Drone Detection System - macOS Launcher"
echo "========================================"

# Python sürümünü kontrol et
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python sürümü: $python_version"

# Gerekli minimum sürüm kontrolü (3.8+)
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✓ Python sürümü uygun"
else
    echo "✗ Python 3.8+ gerekli. Lütfen Python'ı güncelleyin."
    exit 1
fi

# Virtual environment kontrolü
if [ ! -d "venv" ]; then
    echo "Virtual environment oluşturuluyor..."
    python3 -m venv venv
fi

# Virtual environment'ı aktifleştir
source venv/bin/activate

# Paket gereksinimlerini kontrol et ve kur
echo "Bağımlılıklar kontrol ediliyor..."
pip install --upgrade pip

# macOS için özel gereksinimler
if [ ! -f ".mac_deps_installed" ]; then
    echo "macOS bağımlılıkları kuruluyor..."
    
    # OpenCV macOS optimizasyonu
    pip install opencv-python-headless
    
    # Tkinter desteği (homebrew python için)
    if command -v brew &> /dev/null; then
        echo "Homebrew Python kullanılıyor, Tkinter desteği kontrol ediliyor..."
        python3 -c "import tkinter" 2>/dev/null || {
            echo "Tkinter kurulumu gerekli. Homebrew ile python-tk kuruluyor..."
            brew install python-tk
        }
    fi
    
    # İşaret dosyası oluştur
    touch .mac_deps_installed
fi

# Gereksinimler dosyasını kur
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt bulunamadı. Temel bağımlılıklar kuruluyor..."
    pip install opencv-python pillow numpy ultralytics torch torchvision
fi

# macOS için display ayarları
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Retina display desteği
if [[ $(system_profiler SPDisplaysDataType | grep -i retina) ]]; then
    echo "Retina display tespit edildi - UI optimizasyonları aktif"
    export DISPLAY_SCALE=2.0
else
    export DISPLAY_SCALE=1.0
fi

echo "========================================"
echo "Uygulama başlatılıyor..."
echo "macOS optimizasyonları aktif"
echo "========================================"

# Uygulamayı başlat
cd src
python3 main.py

# Virtual environment'tan çık
deactivate