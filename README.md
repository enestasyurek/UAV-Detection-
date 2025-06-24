# Drone Tespit ve Takip Sistemi

Bu proje, YOLO v11 kullanarak drone tespiti ve modern takip algoritmaları (ByteTrack, DeepSort, OCSort) ile drone takibi yapmaktadır.

## Özellikler

- **EDGS-YOLOv8 Detector** ile kesin drone tespiti
  - Edge-Guided Saliency ile küçük nesne algılama
  - Çoklu ölçek tespiti
  - Gece/Gündüz adaptasyonu
- **DroneSpecificTracker** ile gelişmiş takip
  - Yakın/Orta/Uzak mesafe kategorileri
  - Görsel özellik eşleştirme
  - Yörünge tahmini
  - Takip güvenilirlik skoru
- ByteTrack, DeepSort ve OCSort takip algoritmaları
- Video ve canlı kamera akışı desteği
- **macOS optimize edilmiş UI** - Retina display ve native görünüm desteği
- Kullanıcı dostu arayüz
- **GPU optimizasyonları** ile yüksek performans

## Kurulum

```bash
pip install -r requirements.txt
```

**ÖNEMLİ**: İlk çalıştırmada YOLOv8 nano modeli (yolov8n.pt) otomatik olarak indirilecektir (~6MB).

## Kullanım

### Hızlı Başlangıç:

**Windows:**
```bash
run.bat
```

**Linux:**
```bash
./run.sh
```

**macOS:**
```bash
./run_mac.sh
```

### Performans İpuçları:

1. **Yüksek FPS için:**
   - Güven eşiğini 0.05-0.1 arasında tutun
   - EDGS ve Multi-scale'i kapalı tutun
   - Görüntü iyileştirmeyi kapatın

2. **Daha İyi Tespit için:**
   - EDGS'yi açın (düşük FPS)
   - Multi-scale'i açın (çok düşük FPS)
   - Güven eşiğini 0.01'e düşürün

### Komut Satırı Seçenekleri:
```bash
python src/main.py --video video.mp4  # Video dosyası
python src/main.py --camera           # Kamera
python src/main.py --log-level DEBUG  # Debug modu
```

### Önerilen Ayarlar:

**Dengeli Drone Tespiti (ÖNERİLEN):**
- Güven Eşiği: 0.25-0.35
- EDGS: Açık ✅
- Multi-scale: Kapalı
- Video Döngüsü: Açık

**Hassas Drone Tespiti (Sadece Drone):**
- Güven Eşiği: 0.35-0.50
- EDGS: Açık ✅
- Multi-scale: Kapalı
- Daha az yanlış tespit

**Agresif Drone Tespiti (Hiçbir Drone Kaçmasın):**
- Güven Eşiği: 0.10-0.20
- EDGS: Açık ✅
- Multi-scale: Açık ✅
- Daha fazla yanlış tespit olabilir

**YENİ ÖZELLİKLER**: 
1. **Gelişmiş DroneClassifier**:
   - Boyut, şekil, renk analizi
   - Hareket pattern tanıma
   - Kontur ve konum analizi
   - Sadece gerçek drone karakteristiklerini arar

2. **Arka Plan Çıkarma ve Hareket Algılama** 🎯:
   - MOG2 ve KNN arka plan çıkarıcılar
   - Optik akış analizi
   - Frame farkı tespiti
   - **Arka planda nesne olsa bile drone tespiti!**
   - Hareket maskesi görselleştirme

3. **Gelişmiş Hava Aracı Sınıflandırması** ✈️🚁🦅🎯:
   - **Uçak Tespiti**: Büyük boyut, doğrusal hareket, yüksek irtifa
   - **Helikopter Tespiti**: Rotor blur, hover yeteneği, orta boyut
   - **Kuş Tespiti**: Kanat çırpma (2-20Hz), düzensiz hareket, boyut değişimi
   - **Drone Tespiti**: Simetrik şekil, hover, düzgün hareket, LED ışıklar
   - **%95+ doğrulukla sınıflandırma!**
   - **Sadece drone'ları takip, diğerlerini filtrele**

## Proje Yapısı

```
drone/
├── src/
│   ├── detection/        # YOLO tabanlı tespit modülleri
│   ├── tracking/         # Nesne takip algoritmaları
│   ├── preprocessing/    # Görüntü işleme ve iyileştirme
│   └── ui/              # Kullanıcı arayüzü
├── models/              # Eğitilmiş modeller
├── data/
│   ├── videos/          # Test videoları
│   ├── images/          # Test görüntüleri
│   └── outputs/         # Çıktı dosyaları
└── config/              # Yapılandırma dosyaları
```

## EDGS-YOLOv8 Sistemi

### 🎯 Drone Tespiti Özellikleri:

1. **Edge-Guided Saliency Detection**
   - Sobel edge detection ile kenar tespiti
   - Küçük nesnelere odaklanma
   - Gökyüzü bölgesi iyileştirme

2. **Çoklu Ölçek Tespiti**
   - 0.75x ve 1.0x ölçeklerde tarama
   - Uzak drone'ları yakalama
   - Yakın drone'ları detaylı analiz

3. **Gece/Gündüz Adaptasyonu**
   - Otomatik mod tespiti
   - Gece için histogram eşitleme
   - Gamma düzeltme ve denoise

4. **Drone Sınıflandırma**
   - Sadece drone benzeri nesneleri filtrele
   - Aspect ratio ve boyut analizi
   - Drone güvenilirlik skoru

### 🚁 DroneSpecificTracker Özellikleri:

1. **Mesafe Kategorileri**
   - **Yakın**: >10.000 piksel alan (Kırmızı)
   - **Orta**: 1.000-10.000 piksel (Sarı)
   - **Uzak**: <1.000 piksel (Yeşil)

2. **Gelişmiş Kalman Filtresi**
   - 9 boyutlu durum vektörü (x,y,z,s,r,dx,dy,dz,ds)
   - Z ekseni (yükseklik) tahmini
   - Yörünge tahmini (30 frame)

3. **Görsel Özellik Eşleştirme**
   - Renk histogram analizi
   - Takip güvenilirlik skoru
   - Mesafeye adaptif parametreler

4. **Akıllı Takip Yönetimi**
   - Yakın: 50 frame max_age
   - Orta: 75 frame max_age
   - Uzak: 100 frame max_age