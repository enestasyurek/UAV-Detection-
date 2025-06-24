#!/bin/bash

# Linux için Drone Detection System başlatma scripti
echo "Drone Detection System - Linux Launcher"
echo "======================================="

# Platform kontrolü
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS tespit edildi. Lütfen run_mac.sh scriptini kullanın."
    exit 1
fi

# Python sürümünü kontrol et
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 bulunamadı. Lütfen Python 3.8+ yükleyin."
    echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "Arch: sudo pacman -S python python-pip"
    exit 1
fi

python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python sürümü: $python_version"

# Gerekli minimum sürüm kontrolü (3.8+)
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✓ Python sürümü uygun"
else
    echo "✗ Python 3.8+ gerekli. Lütfen Python'ı güncelleyin."
    exit 1
fi

# X11 display kontrolü
if [ -z "$DISPLAY" ]; then
    echo "⚠ DISPLAY değişkeni ayarlanmamış. GUI için X11 gerekli."
    echo "SSH ile bağlanıyorsanız: ssh -X kullanın"
    echo "Yerel terminal için: export DISPLAY=:0"
    read -p "Devam etmek istiyor musunuz? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Sistem bağımlılıklarını kontrol et
echo "Sistem bağımlılıkları kontrol ediliyor..."

# Tkinter kontrolü
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "⚠ Tkinter kurulu değil. Kuruluyor..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3-tk
    elif command -v yum &> /dev/null; then
        sudo yum install -y tkinter
    elif command -v pacman &> /dev/null; then
        sudo pacman -S tk
    else
        echo "✗ Paket yöneticisi tanınmadı. Lütfen python3-tk paketini manuel olarak kurun."
        exit 1
    fi
fi

# OpenCV sistem bağımlılıkları
if ! ldconfig -p | grep -q libgtk; then
    echo "⚠ GTK kütüphaneleri bulunamadı. OpenCV GUI için gerekli."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y libgtk-3-dev libglib2.0-dev
    fi
fi

# Virtual environment oluştur (eğer yoksa)
if [ ! -d "venv" ]; then
    echo "Virtual environment oluşturuluyor..."
    python3 -m venv venv
fi

# Virtual environment'ı aktifleştir
source venv/bin/activate

# Pip güncellemesi
echo "Pip güncelleniyor..."
pip install --upgrade pip

# Linux için özel gereksinimler
if [ ! -f ".linux_deps_installed" ]; then
    echo "Linux bağımlılıkları kuruluyor..."
    
    # OpenCV Linux optimizasyonu
    pip install opencv-python-headless
    
    # İşaret dosyası oluştur
    touch .linux_deps_installed
fi

# Gereksinimler dosyasını kur
if [ -f "requirements.txt" ]; then
    echo "Bağımlılıklar kuruluyor..."
    pip install -r requirements.txt
else
    echo "requirements.txt bulunamadı. Temel bağımlılıklar kuruluyor..."
    pip install opencv-python pillow numpy ultralytics torch torchvision
fi

# Python path'i ayarla
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Linux için çevre değişkenleri
export QT_X11_NO_MITSHM=1  # Qt uyumluluk sorunu çözümü
export OPENCV_VIDEOIO_PRIORITY_V4L2=0  # V4L2 öncelik ayarı

echo "======================================="
echo "Uygulama başlatılıyor..."
echo "Linux optimizasyonları aktif"
echo "======================================="

# Uygulamayı başlat
cd src
python3 main.py
EXIT_CODE=$?

# Virtual environment'tan çık
deactivate

# Sonuç
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================="
    echo "Uygulama başarıyla kapatıldı."
    echo "======================================="
else
    echo "======================================="
    echo "Uygulama hata ile kapatıldı (Kod: $EXIT_CODE)"
    echo "======================================="
    echo ""
    echo "Sorun giderme:"
    echo "1. X11 forwarding: ssh -X"
    echo "2. Display ayarı: export DISPLAY=:0"
    echo "3. Tkinter kurulumu: sudo apt-get install python3-tk"
    echo "4. Kamera izinleri: sudo usermod -a -G video $USER"
    echo ""
fi

exit $EXIT_CODE