# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# Loglama için logging modülünü içe aktar
import logging
# Collections modülünden deque'yi içe aktar
from collections import deque

# Logger oluştur
logger = logging.getLogger(__name__)


class BackgroundSubtractor:
    """Arka plan çıkarma ve hareket algılama sınıfı"""
    
    def __init__(self, history_length: int = 30):
        """
        Arka plan çıkarıcı başlatıcı
        
        Args:
            history_length: Hareket geçmişi uzunluğu
        """
        # MOG2 arka plan çıkarıcı
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        # KNN arka plan çıkarıcı (alternatif)
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN(
            detectShadows=True,
            dist2Threshold=400.0,
            history=500
        )
        
        # Frame geçmişi
        self.frame_history = deque(maxlen=history_length)
        self.motion_masks = deque(maxlen=5)
        
        # Optik akış için
        self.prev_gray = None
        self.use_optical_flow = False  # Performans için kapalı başlat
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Adaptif eşik parametreleri
        self.motion_threshold = 25
        self.min_motion_area = 50
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Frame'i işle ve hareket bilgilerini çıkar
        
        Args:
            frame: BGR görüntü
            
        Returns:
            İşlenmiş görüntüler ve hareket maskeleri
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur uygula (gürültü azaltma)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # MOG2 ile arka plan çıkarma
        fg_mask_mog2 = self.bg_subtractor.apply(blurred)
        
        # KNN ile arka plan çıkarma
        fg_mask_knn = self.knn_subtractor.apply(blurred)
        
        # Maskeleri birleştir
        combined_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Optik akış hesapla (eğer aktifse)
        if self.use_optical_flow:
            optical_flow_mask = self._compute_optical_flow(gray)
        else:
            optical_flow_mask = np.zeros_like(gray)
        
        # Frame farkı hesapla
        frame_diff_mask = self._compute_frame_difference(gray)
        
        # Tüm maskeleri birleştir
        final_motion_mask = self._combine_motion_masks(
            combined_mask, optical_flow_mask, frame_diff_mask
        )
        
        # Final mask boyut kontrolü
        if final_motion_mask.shape[:2] != gray.shape[:2]:
            final_motion_mask = cv2.resize(final_motion_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Hareket bölgelerini bul
        motion_regions = self._find_motion_regions(final_motion_mask)
        
        # Kontrast iyileştirilmiş görüntü
        enhanced_frame = self._enhance_contrast(frame, final_motion_mask)
        
        # Frame geçmişini güncelle
        self.frame_history.append(gray)
        self.motion_masks.append(final_motion_mask)
        
        return {
            'motion_mask': final_motion_mask,
            'motion_regions': motion_regions,
            'enhanced_frame': enhanced_frame,
            'fg_mask_mog2': fg_mask_mog2,
            'fg_mask_knn': fg_mask_knn,
            'optical_flow_mask': optical_flow_mask
        }
        
    def _compute_optical_flow(self, gray: np.ndarray) -> np.ndarray:
        """Optik akış hesapla"""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray)
            
        # Boyut kontrolü - önemli!
        if self.prev_gray.shape != gray.shape:
            # Boyutlar farklıysa, yeniden boyutlandır
            self.prev_gray = cv2.resize(self.prev_gray, (gray.shape[1], gray.shape[0]))
            
        try:
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Akış büyüklüğünü hesapla
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Hareket maskesi oluştur
            motion_mask = (magnitude > self.motion_threshold).astype(np.uint8) * 255
            
        except cv2.error as e:
            # Hata durumunda boş maske dön
            logger.warning(f"Optical flow error: {e}")
            motion_mask = np.zeros_like(gray)
            
        self.prev_gray = gray.copy()
        
        return motion_mask
        
    def _compute_frame_difference(self, gray: np.ndarray) -> np.ndarray:
        """Frame farkı hesapla"""
        if len(self.frame_history) < 2:
            return np.zeros_like(gray)
            
        # Son frame'i al ve boyut kontrolü yap
        prev_frame = self.frame_history[-1]
        if prev_frame.shape != gray.shape:
            prev_frame = cv2.resize(prev_frame, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        # Son iki frame arasındaki fark
        diff = cv2.absdiff(prev_frame, gray)
        
        # Eşikleme
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        return motion_mask
        
    def _combine_motion_masks(self, bg_mask: np.ndarray, 
                            optical_mask: np.ndarray, 
                            diff_mask: np.ndarray) -> np.ndarray:
        """Hareket maskelerini birleştir"""
        # Boyut kontrolü ve düzeltme
        h, w = bg_mask.shape[:2]
        
        # Optical mask boyutunu kontrol et ve gerekirse yeniden boyutlandır
        if optical_mask.shape[:2] != (h, w):
            optical_mask = cv2.resize(optical_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # Diff mask boyutunu kontrol et ve gerekirse yeniden boyutlandır
        if diff_mask.shape[:2] != (h, w):
            diff_mask = cv2.resize(diff_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Ağırlıklı birleştirme
        combined = cv2.addWeighted(bg_mask, 0.4, optical_mask, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.7, diff_mask, 0.3, 0)
        
        # Eşikleme
        _, final_mask = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)
        
        # Küçük gürültüleri temizle
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask
        
    def _find_motion_regions(self, motion_mask: np.ndarray) -> List[Dict]:
        """Hareket bölgelerini bul"""
        # Konturları bul
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Minimum alan kontrolü
            if area < self.min_motion_area:
                continue
                
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Merkez
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
                
            motion_regions.append({
                'bbox': [x, y, x + w, y + h],
                'center': (cx, cy),
                'area': area,
                'contour': contour,
                'aspect_ratio': w / h if h > 0 else 1.0
            })
            
        return motion_regions
        
    def _enhance_contrast(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Hareket bölgelerinde kontrast iyileştirme"""
        enhanced = frame.copy()
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE uygula
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Birleştir
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced_global = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Hareket maskesini 3 kanala genişlet
        motion_mask_3ch = cv2.merge([motion_mask, motion_mask, motion_mask])
        
        # Boyut kontrolü
        if motion_mask_3ch.shape[:2] != enhanced.shape[:2]:
            motion_mask_3ch = cv2.resize(motion_mask_3ch, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Hareket bölgelerinde güçlendirilmiş kontrast
        enhanced = cv2.addWeighted(
            enhanced, 0.3,
            cv2.bitwise_and(enhanced_global, motion_mask_3ch), 0.7,
            0
        )
        
        # Edge enhancement hareket bölgelerinde
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_colored = cv2.merge([edges, edges, edges])
        
        # Edge'leri ekle
        # Boyut kontrolü
        if edges_colored.shape[:2] != motion_mask_3ch.shape[:2]:
            edges_colored = cv2.resize(edges_colored, (motion_mask_3ch.shape[1], motion_mask_3ch.shape[0]), interpolation=cv2.INTER_LINEAR)
        enhanced = cv2.add(enhanced, cv2.bitwise_and(edges_colored, motion_mask_3ch) // 4)
        
        return enhanced
        
    def get_static_background(self) -> Optional[np.ndarray]:
        """Statik arka planı al"""
        if len(self.frame_history) < 10:
            return None
            
        # Median frame (arka plan tahmini)
        frames = np.array(list(self.frame_history))
        background = np.median(frames, axis=0).astype(np.uint8)
        
        return background
        
    def reset(self):
        """Arka plan modelini sıfırla"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN(
            detectShadows=True,
            dist2Threshold=400.0,
            history=500
        )
        self.frame_history.clear()
        self.motion_masks.clear()
        self.prev_gray = None