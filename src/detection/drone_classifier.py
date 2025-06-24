# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple
# Loglama için logging modülünü içe aktar
import logging

# Logger oluştur
logger = logging.getLogger(__name__)


class DroneClassifier:
    """Gelişmiş drone sınıflandırıcı - sadece gerçek drone'ları tespit et"""
    
    def __init__(self):
        """Drone sınıflandırıcı başlatıcı"""
        # Drone karakteristikleri
        self.drone_features = {
            'min_area': 20,  # Minimum alan (piksel)
            'max_area': 50000,  # Maximum alan (piksel)
            'min_aspect_ratio': 0.4,  # Minimum en/boy oranı
            'max_aspect_ratio': 2.5,  # Maximum en/boy oranı
            'min_solidity': 0.5,  # Minimum katılık (dolu/konveks hull)
            'max_perimeter_ratio': 5.0,  # Maximum çevre/alan oranı
            'typical_colors': [  # Tipik drone renkleri (BGR)
                [20, 20, 20],    # Siyah
                [100, 100, 100], # Gri
                [200, 200, 200], # Beyaz
                [50, 50, 150],   # Kırmızı
            ]
        }
        
        # Hareket geçmişi
        self.motion_history = {}
        self.frame_count = 0
        
    def classify_drone(self, frame: np.ndarray, detection: Dict) -> Tuple[bool, float]:
        """
        Tespit edilen nesnenin drone olup olmadığını belirle
        
        Args:
            frame: Görüntü frame'i
            detection: Tespit bilgileri
            
        Returns:
            (is_drone, confidence) tuple'ı
        """
        scores = {}
        
        # 1. Boyut analizi
        scores['size'] = self._analyze_size(detection)
        
        # 2. Şekil analizi
        scores['shape'] = self._analyze_shape(frame, detection)
        
        # 3. Renk analizi
        scores['color'] = self._analyze_color(frame, detection)
        
        # 4. Hareket analizi
        scores['motion'] = self._analyze_motion_pattern(detection)
        
        # 5. Konum analizi
        scores['position'] = self._analyze_position(frame, detection)
        
        # 6. Kontur analizi
        scores['contour'] = self._analyze_contour(frame, detection)
        
        # Ağırlıklı ortalama
        weights = {
            'size': 0.20,
            'shape': 0.25,
            'color': 0.15,
            'motion': 0.20,
            'position': 0.10,
            'contour': 0.10
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        # Drone kararı
        is_drone = total_score > 0.4  # Eşik değeri
        
        # Debug log
        if total_score > 0.3:
            logger.debug(f"Drone score: {total_score:.2f}, Details: {scores}")
        
        return is_drone, total_score
        
    def _analyze_size(self, detection: Dict) -> float:
        """Boyut analizi"""
        area = detection.get('area', 0)
        
        # Drone boyut aralığı kontrolü
        if self.drone_features['min_area'] <= area <= self.drone_features['max_area']:
            # Tipik drone boyutlarına yakınlık
            if 100 <= area <= 5000:
                return 1.0  # Mükemmel boyut
            elif 50 <= area <= 10000:
                return 0.7  # İyi boyut
            else:
                return 0.4  # Kabul edilebilir
        return 0.1  # Drone için uygun değil
        
    def _analyze_shape(self, frame: np.ndarray, detection: Dict) -> float:
        """Şekil analizi"""
        aspect_ratio = detection.get('aspect_ratio', 1.0)
        
        # Drone şekil kontrolü
        if self.drone_features['min_aspect_ratio'] <= aspect_ratio <= self.drone_features['max_aspect_ratio']:
            # Kare veya hafif dikdörtgen (tipik drone)
            if 0.7 <= aspect_ratio <= 1.3:
                return 1.0
            else:
                return 0.6
        return 0.2
        
    def _analyze_color(self, frame: np.ndarray, detection: Dict) -> float:
        """Renk analizi"""
        x1, y1, x2, y2 = detection['bbox']
        
        # Sınırları kontrol et
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # ROI al
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.5  # Nötr
            
        # Ortalama renk
        mean_color = roi.mean(axis=(0, 1))
        
        # Gri tonlama kontrolü (çoğu drone siyah/beyaz/gri)
        color_std = np.std(mean_color)
        if color_std < 30:  # Düşük renk varyansı = gri tonlar
            return 0.8
            
        # Tipik drone renklerine yakınlık
        min_dist = float('inf')
        for typical_color in self.drone_features['typical_colors']:
            dist = np.linalg.norm(mean_color - typical_color)
            min_dist = min(min_dist, dist)
            
        if min_dist < 50:
            return 0.7
        elif min_dist < 100:
            return 0.4
        else:
            return 0.2
            
    def _analyze_motion_pattern(self, detection: Dict) -> float:
        """Hareket pattern analizi"""
        track_id = detection.get('track_id', id(detection))
        center = detection['center']
        
        # Hareket geçmişini güncelle
        if track_id not in self.motion_history:
            self.motion_history[track_id] = []
            
        self.motion_history[track_id].append(center)
        
        # Son 10 frame'i tut
        if len(self.motion_history[track_id]) > 10:
            self.motion_history[track_id].pop(0)
            
        # Hareket analizi
        if len(self.motion_history[track_id]) < 3:
            return 0.5  # Yeterli veri yok
            
        # Hareket tutarlılığı
        positions = self.motion_history[track_id]
        velocities = []
        for i in range(1, len(positions)):
            vx = positions[i][0] - positions[i-1][0]
            vy = positions[i][1] - positions[i-1][1]
            velocities.append((vx, vy))
            
        # Hız tutarlılığı (drone'lar genelde düzgün hareket eder)
        if velocities:
            vx_std = np.std([v[0] for v in velocities])
            vy_std = np.std([v[1] for v in velocities])
            
            # Düşük standart sapma = tutarlı hareket
            if vx_std < 5 and vy_std < 5:
                return 0.9  # Çok tutarlı
            elif vx_std < 10 and vy_std < 10:
                return 0.6  # Tutarlı
            else:
                return 0.3  # Tutarsız
                
        return 0.5
        
    def _analyze_position(self, frame: np.ndarray, detection: Dict) -> float:
        """Konum analizi"""
        h, w = frame.shape[:2]
        cx, cy = detection['center']
        
        # Gökyüzü bölgesi (üst %60)
        if cy < h * 0.6:
            return 0.8
        # Orta bölge
        elif cy < h * 0.8:
            return 0.5
        # Alt bölge (drone'lar nadiren yerde)
        else:
            return 0.2
            
    def _analyze_contour(self, frame: np.ndarray, detection: Dict) -> float:
        """Kontur analizi"""
        x1, y1, x2, y2 = detection['bbox']
        
        # ROI al
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.5
            
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Kontur bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.3
            
        # En büyük konturu al
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Kontur özellikleri
        area = cv2.contourArea(largest_contour)
        if area > 0:
            # Konveks hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            # Solidity (katılık)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Drone'lar genelde yüksek solidity'ye sahip
            if solidity > self.drone_features['min_solidity']:
                return 0.8
            else:
                return 0.4
                
        return 0.5
        
    def update_frame_count(self):
        """Frame sayacını güncelle"""
        self.frame_count += 1
        
        # Eski hareket geçmişlerini temizle
        if self.frame_count % 100 == 0:
            # 30 frame'den eski olanları sil
            to_remove = []
            for track_id in self.motion_history:
                if len(self.motion_history[track_id]) == 0:
                    to_remove.append(track_id)
            for track_id in to_remove:
                del self.motion_history[track_id]