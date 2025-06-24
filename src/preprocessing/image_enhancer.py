# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Tip belirtimi için gerekli modülleri içe aktar
from typing import Tuple, Optional, Dict
# Enum için gerekli modülü içe aktar
from enum import Enum
# Loglama için logging modülünü içe aktar
import logging

# Logger oluştur
logger = logging.getLogger(__name__)


class CameraMode(Enum):
    """Kamera modu tanımlamaları"""
    DAY = "day"          # Gündüz modu
    NIGHT = "night"      # Gece modu
    AUTO = "auto"        # Otomatik mod seçimi


class ImageEnhancer:
    """Gece ve gündüz görüntü iyileştirme sınıfı"""
    
    def __init__(self):
        """Görüntü iyileştirici başlatıcı"""
        # Varsayılan parametreler
        self.brightness_threshold = 60  # Gece/gündüz ayırma eşiği
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Drone tespiti için özel parametreler
        self.drone_enhancement_enabled = False  # Performans için kapalı
        self.sky_detection_threshold = 200  # Gökyüzü parlaklık eşiği
        self.fast_mode = True  # Hızlı mod
        
    def enhance_image(self, image: np.ndarray, mode: CameraMode = CameraMode.AUTO) -> np.ndarray:
        """
        Görüntüyü belirtilen moda göre iyileştir
        
        Args:
            image: Giriş görüntüsü (BGR)
            mode: Kamera modu
            
        Returns:
            İyileştirilmiş görüntü
        """
        # Hızlı mod - minimal işleme
        if self.fast_mode:
            # Sadece basit kontrast ayarı
            return cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            
        # Otomatik mod tespiti
        if mode == CameraMode.AUTO:
            mode = self._detect_mode(image)
            logger.debug(f"Auto-detected mode: {mode.value}")
            
        # Moda göre iyileştirme uygula
        if mode == CameraMode.NIGHT:
            return self._enhance_night_image(image)
        else:
            return self._enhance_day_image(image)
            
    def _detect_mode(self, image: np.ndarray) -> CameraMode:
        """
        Görüntünün gece mi gündüz mü olduğunu tespit et
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Tespit edilen mod
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ortalama parlaklığı hesapla
        mean_brightness = np.mean(gray)
        
        # Eşik kontrolü
        if mean_brightness < self.brightness_threshold:
            return CameraMode.NIGHT
        else:
            return CameraMode.DAY
            
    def _enhance_night_image(self, image: np.ndarray) -> np.ndarray:
        """
        Gelişmiş gece görüntüsü iyileştirme - PERFORMANS OPTİMİZE
        
        Args:
            image: Gece görüntüsü
            
        Returns:
            İyileştirilmiş görüntü
        """
        # Kopyasını oluştur
        enhanced = image.copy()
        
        # 1. LAB renk uzayında CLAHE (en etkili)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Daha agresif CLAHE parametreleri
        night_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_channel = night_clahe.apply(l_channel)
        
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. Gamma düzeltmesi (daha güçlü)
        enhanced = self._apply_gamma_correction(enhanced, gamma=2.0)
        
        # 3. Basit denoise (hızlı)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # 4. Hafif keskinleştirme
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    def _enhance_day_image(self, image: np.ndarray) -> np.ndarray:
        """
        Gündüz görüntüsünü iyileştir
        
        Args:
            image: Gündüz görüntüsü
            
        Returns:
            İyileştirilmiş görüntü
        """
        # Kopyasını oluştur
        enhanced = image.copy()
        
        # 1. Otomatik kontrast ve parlaklık ayarı
        enhanced = self._auto_brightness_contrast(enhanced)
        
        # 2. Hafif keskinlik artırma
        enhanced = self._sharpen_image(enhanced, strength=0.3)
        
        # 3. Renk canlılığı artırma
        enhanced = self._enhance_saturation(enhanced, factor=1.1)
        
        # 4. Parlak alanları dengele
        enhanced = self._reduce_highlights(enhanced)
        
        return enhanced
        
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Gamma düzeltmesi uygula
        
        Args:
            image: Giriş görüntüsü
            gamma: Gamma değeri (>1 parlaklaştırır, <1 koyulaştırır)
            
        Returns:
            Düzeltilmiş görüntü
        """
        # Arama tablosu oluştur
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        # Tabloya göre dönüştür
        return cv2.LUT(image, table)
        
    def _sharpen_image(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Görüntüyü keskinleştir
        
        Args:
            image: Giriş görüntüsü
            strength: Keskinleştirme gücü (0-1)
            
        Returns:
            Keskinleştirilmiş görüntü
        """
        # Gaussian blur uygula
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        
        return sharpened
        
    def _auto_brightness_contrast(self, image: np.ndarray, 
                                 clip_hist_percent: float = 1.0) -> np.ndarray:
        """
        Otomatik parlaklık ve kontrast ayarı
        
        Args:
            image: Giriş görüntüsü
            clip_hist_percent: Histogram kırpma yüzdesi
            
        Returns:
            Ayarlanmış görüntü
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Histogram hesapla
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)
        
        # Kümülatif dağılım
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))
            
        # Maksimum değer
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0
        
        # Minimum ve maksimum gri seviyeyi bul
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
            
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
            
        # Alpha ve beta hesapla
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        
        # Uygula
        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return auto_result
        
    def _color_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Renk dengeleme uygula
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Renk dengelenmiş görüntü
        """
        # Her kanal için histogram eşitleme
        result = np.zeros_like(image)
        
        for i in range(3):  # BGR kanalları
            # Histogram eşitleme
            result[:, :, i] = cv2.equalizeHist(image[:, :, i])
            
        # Orijinal ile karıştır (çok agresif olmaması için)
        result = cv2.addWeighted(image, 0.5, result, 0.5, 0)
        
        return result
        
    def _enhance_saturation(self, image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Renk doygunluğunu artır
        
        Args:
            image: Giriş görüntüsü
            factor: Doygunluk faktörü (1.0 = değişim yok)
            
        Returns:
            Doygunluğu artırılmış görüntü
        """
        # HSV'ye çevir
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Doygunluk kanalını artır
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        
        # Sınırları kontrol et
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        
        # BGR'ye geri çevir
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
        
    def _reduce_highlights(self, image: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        Parlak alanları azalt
        
        Args:
            image: Giriş görüntüsü
            threshold: Parlak alan eşiği
            
        Returns:
            Parlak alanları azaltılmış görüntü
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Parlak alanları bul
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Maske genişlet
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        
        # Parlak alanları azalt
        result = image.copy()
        result[bright_mask > 0] = (result[bright_mask > 0] * 0.7).astype(np.uint8)
        
        # Yumuşat
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result
        
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Tespit için görüntüyü ön işle
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Ön işlenmiş görüntü
        """
        # Önce iyileştir
        enhanced = self.enhance_image(image)
        
        # Drone tespiti için özel ön işleme
        if self.drone_enhancement_enabled:
            enhanced = self._enhance_for_drones(enhanced)
        
        # Tespit için ek ön işleme
        # 1. Kenar iyileştirme
        enhanced = self._enhance_edges(enhanced)
        
        # 2. Morfolojik işlemler
        enhanced = self._apply_morphology(enhanced)
        
        return enhanced
        
    def _enhance_for_drones(self, image: np.ndarray) -> np.ndarray:
        """
        Drone tespiti için özel iyileştirmeler
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Drone tespiti için optimize edilmiş görüntü
        """
        # Gökyüzü bölgelerini tespit et ve iyileştir
        enhanced = image.copy()
        
        # HSV'ye çevir
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        # Gökyüzü maskesi (açık mavi tonları)
        lower_sky = np.array([85, 30, 30])  # Düşük doygunluk, açık mavi
        upper_sky = np.array([135, 255, 255])
        sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
        
        # Beyaz/gri gökyüzü için ek maske
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, self.sky_detection_threshold, 255, cv2.THRESH_BINARY)
        
        # Maskeleri birleştir
        sky_mask = cv2.bitwise_or(sky_mask, bright_mask)
        
        # Gökyüzü bölgelerinde kontrast artırma
        sky_region = cv2.bitwise_and(enhanced, enhanced, mask=sky_mask)
        
        # Gökyüzü bölgesi için özel CLAHE
        sky_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        
        # Her kanal için ayrı işlem
        for i in range(3):
            channel = enhanced[:, :, i]
            sky_channel = sky_clahe.apply(channel)
            enhanced[:, :, i] = np.where(sky_mask > 0, sky_channel, channel)
            
        # Küçük nesneler için keskinleştirme
        # Unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Gürültü azaltma (küçük drone'ları kaybetmemek için hafif)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        return enhanced
        
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Kenarları iyileştir
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Kenarları iyileştirilmiş görüntü
        """
        # Laplacian filtresi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Kenarları orijinal görüntüye ekle
        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.add(result[:, :, i], laplacian // 3)
            
        return result
        
    def _apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Morfolojik işlemler uygula
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            İşlenmiş görüntü
        """
        # Morfolojik gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        # Orijinal ile karıştır
        result = cv2.addWeighted(image, 0.8, gradient, 0.2, 0)
        
        return result
        
    def get_enhancement_stats(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """
        İyileştirme istatistiklerini hesapla
        
        Args:
            original: Orijinal görüntü
            enhanced: İyileştirilmiş görüntü
            
        Returns:
            İstatistikler sözlüğü
        """
        # Ortalama parlaklık
        orig_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        enh_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        
        # Kontrast (standart sapma)
        orig_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        enh_contrast = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr = cv2.PSNR(original, enhanced)
        
        return {
            'original_brightness': float(orig_brightness),
            'enhanced_brightness': float(enh_brightness),
            'brightness_change': float(enh_brightness - orig_brightness),
            'original_contrast': float(orig_contrast),
            'enhanced_contrast': float(enh_contrast),
            'contrast_change': float(enh_contrast - orig_contrast),
            'psnr': float(psnr)
        }