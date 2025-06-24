# 🚁 Drone Tespit ve Takip İpuçları

## 🎯 YENİ: Gelişmiş DroneClassifier Sistemi

Sistem artık sadece gerçek drone'ları tespit ediyor! 

### Drone Tespit Kriterleri:
- ✅ Boyut: 20-50,000 piksel arası
- ✅ Şekil: 0.4-2.5 aspect ratio
- ✅ Renk: Siyah, gri, beyaz tonları
- ✅ Hareket: Tutarlı, düzgün
- ✅ Konum: Genelde gökyüzünde
- ✅ **YENİ: Arka Plan Ayrıştırma** - Karmaşık arka planlarda bile tespit!

## ✈️ YENİ: Gelişmiş Hava Aracı Sınıflandırması

### Sistem Artık 4 Tip Hava Aracını Ayırt Eder!

**✈️ Uçak Özellikleri:**
- 📏 Büyük boyut (5000-100000 piksel²)
- ➡️ Doğrusal yörünge (R² > 0.9)
- 🌙 Gece navigation ışıkları
- 📐 Uzun gövde (aspect ratio 2-5)

**🚁 Helikopter Özellikleri:**
- 🔄 Rotor blur etkisi
- ⚓ Hover yeteneği
- 📏 Orta-büyük boyut (10000-50000 piksel²)
- 🎯 10-30 Hz rotor frekansı

**🦅 Kuş Özellikleri:**
- 🪶 Kanat çırpma (2-20 Hz)
- 🌊 Düzensiz hareket paterni
- 📏 Boyut değişimi (%30+)
- 🔄 Asimetrik şekil

**🎯 Drone Özellikleri:**
- ⚓ Hover yeteneği
- ➡️ Düzgün hareket
- 📐 Simetrik şekil
- 💡 LED ışıkları (gece)

## 🆕 Arka Plan Çıkarma Sistemi

### Arka Planda Nesne Sorunu ÇÖZÜLDÜ!
- **MOG2 + KNN** arka plan çıkarıcılar
- **Optik Akış** ile hareket takibi
- **Frame Farkı** analizi
- Hareket eden her nesne tespit edilir
- Statik arka plan otomatik ayrıştırılır

### Hareket Maskesi Görüntüleme:
- UI'da "Hareket Maskesi" seçeneğini işaretleyin
- Yeşil alanlar: Hareket algılanan bölgeler
- Sarı konturlar: Hareket sınırları

## 🎯 En İyi Ayarlar

### Dengeli Tespit (ÖNERİLEN):
- **Güven Eşiği**: 0.25 - 0.35
- **EDGS**: AÇIK ✅
- **Multi-scale**: KAPALI
- **Tracker**: Drone-Specific

### Hassas Tespit (Sadece Kesin Drone):
- **Güven Eşiği**: 0.40 - 0.50
- **EDGS**: AÇIK ✅
- **Multi-scale**: KAPALI
- **Tracker**: Drone-Specific

### Agresif Tespit (Hiçbir Drone Kaçmasın):
- **Güven Eşiği**: 0.10 - 0.20
- **EDGS**: AÇIK ✅
- **Multi-scale**: AÇIK ✅
- **Tracker**: Drone-Specific

## 📹 Video İpuçları

1. **Video Döngüsü**: "Video Döngüsü" seçeneği aktifken video bittiğinde başa döner
2. **Yeni Video**: Durdur → Yeni video seç → Başlat
3. **Takip Sürekliliği**: Sistem kayıp drone'ları 200 frame boyunca takip eder

## 🔧 Sorun Giderme

### Drone Tespit Edilmiyor:
1. Güven eşiğini 0.001'e düşürün
2. EDGS ve Multi-scale'i açın
3. Görüntü İyileştirmeyi açın

### Düşük FPS:
1. EDGS ve Multi-scale'i kapatın
2. Güven eşiğini 0.05'e yükseltin
3. Görüntü İyileştirmeyi kapatın

### Yanlış Tespitler:
- Normal! Sistem agresif ayarlanmış durumda
- Takip algoritması yanlış tespitleri filtreler

## 🎮 Kullanım

1. Programı başlatın: `./run.bat`
2. Video seçin veya kamera açın
3. Ayarları drone tipine göre yapın
4. "Başlat" butonuna basın

## 📊 Performans

- **GPU**: CUDA varsa otomatik kullanır
- **CPU**: GPU yoksa CPU kullanır (yavaş)
- **Hedef FPS**: 30+ (GPU), 10-15 (CPU)

## 🔴 Canlı Test

Test videosunda drone yoksa:
- Herhangi bir küçük hareketli nesne drone olarak algılanır
- Kuşlar, uçaklar, toplar vb. drone olarak görünür
- Bu normaldir - gerçek kullanımda daha iyi çalışır