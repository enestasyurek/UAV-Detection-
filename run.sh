#!/bin/bash

echo "Drone Tespit ve Takip Sistemi Başlatılıyor..."
echo

# Python kontrolü
if ! command -v python3 &> /dev/null; then
    echo "HATA: Python yüklü değil!"
    echo "Lütfen Python 3.8 veya üzerini yükleyin."
    exit 1
fi

# Sanal ortam kontrolü
if [ ! -d "venv" ]; then
    echo "Sanal ortam oluşturuluyor..."
    python3 -m venv venv
    echo
fi

# Sanal ortamı aktif et
echo "Sanal ortam aktif ediliyor..."
source venv/bin/activate

# Bağımlılıkları yükle
echo
echo "Bağımlılıklar kontrol ediliyor..."
if ! pip install -r requirements.txt &> /dev/null; then
    echo "Bağımlılıklar yükleniyor..."
    pip install -r requirements.txt
fi

# Uygulamayı başlat
echo
echo "Uygulama başlatılıyor..."
echo
python src/main.py "$@"

# Hata kontrolü
if [ $? -ne 0 ]; then
    echo
    echo "HATA: Uygulama çalıştırılırken hata oluştu!"
    read -p "Devam etmek için Enter'a basın..."
fi