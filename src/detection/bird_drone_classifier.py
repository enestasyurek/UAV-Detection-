# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional, Deque
# Loglama için logging modülünü içe aktar
import logging
# Collections modülünden deque'yi içe aktar
from collections import deque
# Math modülünü içe aktar
import math
# SciPy signal processing için
from scipy import signal

# Logger oluştur
logger = logging.getLogger(__name__)


class BirdDroneClassifier:
    """Kuş ve drone ayırımı için gelişmiş sınıflandırıcı"""
    
    def __init__(self):
        """Kuş-Drone sınıflandırıcı başlatıcı"""
        # Nesne geçmişleri
        self.object_histories = {}  # track_id -> özelliklere geçmiş
        self.history_length = 30  # 1 saniyelik geçmiş (30 FPS)
        
        # Kanat çırpma analizi için
        self.wing_flap_min_freq = 2.0   # Hz - minimum kanat çırpma frekansı
        self.wing_flap_max_freq = 20.0  # Hz - maksimum kanat çırpma frekansı
        
        # Drone özellikleri
        self.drone_features = {
            'hover_threshold': 5.0,  # piksel - hover için maksimum hareket
            'motion_smoothness': 0.8,  # hareket düzgünlüğü eşiği
            'size_variance_threshold': 0.15,  # boyut değişim eşiği
            'symmetry_threshold': 0.7,  # simetri skoru eşiği
        }
        
        # Kuş özellikleri
        self.bird_features = {
            'size_change_ratio': 0.3,  # kanat açma/kapama boyut değişimi
            'motion_irregularity': 0.5,  # hareket düzensizliği
            'glide_detection': True,  # süzülme tespiti
        }
        
    def classify(self, frame: np.ndarray, detection: Dict, track_id: Optional[str] = None) -> Tuple[str, float, Dict]:
        """
        Nesnenin kuş mu drone mu olduğunu belirle
        
        Args:
            frame: Görüntü frame'i
            detection: Tespit bilgileri
            track_id: Takip ID'si
            
        Returns:
            (class_name, confidence, features) - 'drone', 'bird' veya 'unknown'
        """
        if track_id is None:
            track_id = str(id(detection))
            
        # Geçmişi güncelle
        self._update_history(track_id, detection, frame)
        
        # Özellikleri analiz et
        features = {}
        
        # 1. Hareket analizi
        motion_features = self._analyze_motion(track_id)
        features.update(motion_features)
        
        # 2. Boyut değişimi analizi
        size_features = self._analyze_size_variation(track_id)
        features.update(size_features)
        
        # 3. Kanat çırpma analizi
        wing_features = self._analyze_wing_flapping(track_id)
        features.update(wing_features)
        
        # 4. Şekil ve simetri analizi
        shape_features = self._analyze_shape(frame, detection)
        features.update(shape_features)
        
        # 5. Hover (asılı kalma) tespiti
        hover_features = self._detect_hovering(track_id)
        features.update(hover_features)
        
        # Skorlama
        drone_score = self._calculate_drone_score(features)
        bird_score = self._calculate_bird_score(features)
        
        # Sınıflandırma
        if drone_score > bird_score and drone_score > 0.6:
            return 'drone', drone_score, features
        elif bird_score > drone_score and bird_score > 0.6:
            return 'bird', bird_score, features
        else:
            return 'unknown', max(drone_score, bird_score), features
            
    def _update_history(self, track_id: str, detection: Dict, frame: np.ndarray):
        """Nesne geçmişini güncelle"""
        if track_id not in self.object_histories:
            self.object_histories[track_id] = {
                'positions': deque(maxlen=self.history_length),
                'sizes': deque(maxlen=self.history_length),
                'aspects': deque(maxlen=self.history_length),
                'timestamps': deque(maxlen=self.history_length),
                'shapes': deque(maxlen=10),  # Şekil örnekleri
            }
            
        history = self.object_histories[track_id]
        
        # Pozisyon
        history['positions'].append(detection['center'])
        
        # Boyut (alan)
        history['sizes'].append(detection['area'])
        
        # En-boy oranı
        history['aspects'].append(detection.get('aspect_ratio', 1.0))
        
        # Zaman damgası
        import time
        history['timestamps'].append(time.time())
        
        # Şekil örneği (her 3 frame'de bir)
        if len(history['positions']) % 3 == 0:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Küçük boyuta getir
                    roi_small = cv2.resize(roi, (32, 32))
                    history['shapes'].append(roi_small)
                    
    def _analyze_motion(self, track_id: str) -> Dict:
        """Hareket pattern analizi"""
        if track_id not in self.object_histories:
            return {}
            
        history = self.object_histories[track_id]
        positions = list(history['positions'])
        
        if len(positions) < 5:
            return {}
            
        features = {}
        
        # Hız hesapla
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speed = math.sqrt(dx**2 + dy**2)
            velocities.append(speed)
            
        if velocities:
            # Ortalama hız
            avg_speed = np.mean(velocities)
            features['avg_speed'] = avg_speed
            
            # Hız varyansı (düzgünlük)
            speed_variance = np.var(velocities)
            features['speed_variance'] = speed_variance
            features['motion_smoothness'] = 1.0 / (1.0 + speed_variance)
            
            # Yön değişimleri
            direction_changes = 0
            for i in range(1, len(velocities)):
                if velocities[i] * velocities[i-1] < 0:  # Yön değişimi
                    direction_changes += 1
            features['direction_changes'] = direction_changes
            
        # Yörünge düzgünlüğü
        if len(positions) > 10:
            # Pozisyonları numpy array'e çevir
            pos_array = np.array(positions)
            
            # Polinom fit (2. derece)
            t = np.arange(len(positions))
            poly_x = np.polyfit(t, pos_array[:, 0], 2)
            poly_y = np.polyfit(t, pos_array[:, 1], 2)
            
            # Fit kalitesi
            fitted_x = np.polyval(poly_x, t)
            fitted_y = np.polyval(poly_y, t)
            
            error_x = np.mean(np.abs(pos_array[:, 0] - fitted_x))
            error_y = np.mean(np.abs(pos_array[:, 1] - fitted_y))
            
            trajectory_smoothness = 1.0 / (1.0 + error_x + error_y)
            features['trajectory_smoothness'] = trajectory_smoothness
            
        return features
        
    def _analyze_size_variation(self, track_id: str) -> Dict:
        """Boyut değişimi analizi (kanat çırpma için)"""
        if track_id not in self.object_histories:
            return {}
            
        history = self.object_histories[track_id]
        sizes = list(history['sizes'])
        
        if len(sizes) < 10:
            return {}
            
        features = {}
        
        # Boyut varyansı
        size_variance = np.var(sizes) / (np.mean(sizes) + 1e-6)
        features['size_variance_ratio'] = size_variance
        
        # Periyodik boyut değişimi (kanat çırpma)
        if len(sizes) > 20:
            # Normalize et
            sizes_norm = (sizes - np.mean(sizes)) / (np.std(sizes) + 1e-6)
            
            # FFT ile frekans analizi
            fft = np.fft.fft(sizes_norm)
            freqs = np.fft.fftfreq(len(sizes_norm), d=1/30.0)  # 30 FPS varsayım
            
            # Pozitif frekanslar
            positive_freqs = freqs[:len(freqs)//2]
            fft_magnitude = np.abs(fft[:len(fft)//2])
            
            # Kanat çırpma frekans aralığında peak ara
            wing_freq_mask = (positive_freqs >= self.wing_flap_min_freq) & \
                           (positive_freqs <= self.wing_flap_max_freq)
            
            if np.any(wing_freq_mask):
                wing_band_power = np.max(fft_magnitude[wing_freq_mask])
                total_power = np.sum(fft_magnitude)
                
                features['wing_flap_power'] = wing_band_power / (total_power + 1e-6)
                
                # En güçlü frekansı bul
                peak_idx = np.argmax(fft_magnitude[wing_freq_mask])
                peak_freq = positive_freqs[wing_freq_mask][peak_idx]
                features['dominant_frequency'] = peak_freq
                
        return features
        
    def _analyze_wing_flapping(self, track_id: str) -> Dict:
        """Kanat çırpma pattern analizi"""
        if track_id not in self.object_histories:
            return {}
            
        history = self.object_histories[track_id]
        aspects = list(history['aspects'])
        
        if len(aspects) < 15:
            return {}
            
        features = {}
        
        # En-boy oranı değişimi (kanat açma/kapama)
        aspect_variance = np.var(aspects)
        features['aspect_variance'] = aspect_variance
        
        # Periyodik pattern arama
        if len(aspects) > 20:
            # Autocorrelation
            aspects_norm = (aspects - np.mean(aspects)) / (np.std(aspects) + 1e-6)
            autocorr = np.correlate(aspects_norm, aspects_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Pozitif lag'ler
            
            # İlk birkaç lag'i atla
            autocorr = autocorr[3:]
            
            # Periyodik peak'ler ara
            peaks, properties = signal.find_peaks(autocorr, height=0.3, distance=3)
            
            if len(peaks) > 0:
                # En güçlü periyod
                strongest_peak = peaks[np.argmax(properties['peak_heights'])]
                period = strongest_peak + 3  # Atladığımız lag'leri ekle
                
                flap_frequency = 30.0 / period  # 30 FPS varsayım
                features['flap_frequency'] = flap_frequency
                features['flap_periodicity'] = properties['peak_heights'][np.argmax(properties['peak_heights'])]
                
        return features
        
    def _analyze_shape(self, frame: np.ndarray, detection: Dict) -> Dict:
        """Şekil ve simetri analizi"""
        features = {}
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # ROI al
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return features
            
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Binary mask oluştur
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Kontur bul
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük kontur
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Konveks hull
            hull = cv2.convexHull(largest_contour)
            
            # Solidity (doluluq)
            contour_area = cv2.contourArea(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = contour_area / hull_area
                features['solidity'] = solidity
                
            # Simetri analizi
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Sol-sağ simetri
                left_half = binary[:, :cx]
                right_half = binary[:, cx:]
                
                if right_half.shape[1] > 0:
                    # Sağ yarıyı flip et
                    right_flipped = cv2.flip(right_half, 1)
                    
                    # Boyutları eşitle
                    min_width = min(left_half.shape[1], right_flipped.shape[1])
                    if min_width > 0:
                        left_half = left_half[:, -min_width:]
                        right_flipped = right_flipped[:, -min_width:]
                        
                        # Benzerlik hesapla
                        similarity = np.sum(left_half == right_flipped) / (left_half.size + 1e-6)
                        features['horizontal_symmetry'] = similarity
                        
        return features
        
    def _detect_hovering(self, track_id: str) -> Dict:
        """Hover (asılı kalma) tespiti"""
        if track_id not in self.object_histories:
            return {}
            
        history = self.object_histories[track_id]
        positions = list(history['positions'])
        
        if len(positions) < 10:
            return {}
            
        features = {}
        
        # Son 10 frame'deki hareket
        recent_positions = positions[-10:]
        
        # Ortalama pozisyon
        mean_x = np.mean([p[0] for p in recent_positions])
        mean_y = np.mean([p[1] for p in recent_positions])
        
        # Ortalamadan sapma
        deviations = []
        for px, py in recent_positions:
            dist = math.sqrt((px - mean_x)**2 + (py - mean_y)**2)
            deviations.append(dist)
            
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        
        # Hover skoru
        is_hovering = max_deviation < self.drone_features['hover_threshold']
        hover_stability = 1.0 / (1.0 + avg_deviation)
        
        features['is_hovering'] = is_hovering
        features['hover_stability'] = hover_stability
        features['position_deviation'] = avg_deviation
        
        return features
        
    def _calculate_drone_score(self, features: Dict) -> float:
        """Drone olma skorunu hesapla"""
        score = 0.0
        weights_sum = 0.0
        
        # Hareket düzgünlüğü
        if 'motion_smoothness' in features:
            score += features['motion_smoothness'] * 0.25
            weights_sum += 0.25
            
        # Yörünge düzgünlüğü
        if 'trajectory_smoothness' in features:
            score += features['trajectory_smoothness'] * 0.20
            weights_sum += 0.20
            
        # Hover yeteneği
        if 'hover_stability' in features:
            score += features['hover_stability'] * 0.15
            weights_sum += 0.15
            
        # Düşük boyut varyansı (sabit boyut)
        if 'size_variance_ratio' in features:
            size_stability = 1.0 / (1.0 + features['size_variance_ratio'])
            score += size_stability * 0.15
            weights_sum += 0.15
            
        # Simetri
        if 'horizontal_symmetry' in features:
            score += features['horizontal_symmetry'] * 0.10
            weights_sum += 0.10
            
        # Yüksek solidity
        if 'solidity' in features:
            score += features['solidity'] * 0.10
            weights_sum += 0.10
            
        # Düşük kanat çırpma
        if 'wing_flap_power' in features:
            no_flapping = 1.0 - features['wing_flap_power']
            score += no_flapping * 0.05
            weights_sum += 0.05
            
        # Normalize et
        if weights_sum > 0:
            score = score / weights_sum
            
        return score
        
    def _calculate_bird_score(self, features: Dict) -> float:
        """Kuş olma skorunu hesapla"""
        score = 0.0
        weights_sum = 0.0
        
        # Kanat çırpma
        if 'wing_flap_power' in features:
            score += features['wing_flap_power'] * 0.30
            weights_sum += 0.30
            
        # Boyut varyansı (kanat açma/kapama)
        if 'size_variance_ratio' in features:
            size_variation = min(features['size_variance_ratio'], 1.0)
            score += size_variation * 0.20
            weights_sum += 0.20
            
        # Aspect ratio varyansı
        if 'aspect_variance' in features:
            aspect_variation = min(features['aspect_variance'], 1.0)
            score += aspect_variation * 0.15
            weights_sum += 0.15
            
        # Hareket düzensizliği
        if 'motion_smoothness' in features:
            irregularity = 1.0 - features['motion_smoothness']
            score += irregularity * 0.15
            weights_sum += 0.15
            
        # Yön değişimleri
        if 'direction_changes' in features:
            dir_change_score = min(features['direction_changes'] / 10.0, 1.0)
            score += dir_change_score * 0.10
            weights_sum += 0.10
            
        # Düşük hover yeteneği
        if 'hover_stability' in features:
            no_hover = 1.0 - features['hover_stability']
            score += no_hover * 0.10
            weights_sum += 0.10
            
        # Normalize et
        if weights_sum > 0:
            score = score / weights_sum
            
        return score
        
    def cleanup_old_tracks(self, active_track_ids: List[str]):
        """Eski takipleri temizle"""
        # Aktif olmayan takipleri sil
        to_remove = []
        for track_id in self.object_histories:
            if track_id not in active_track_ids:
                to_remove.append(track_id)
                
        for track_id in to_remove:
            del self.object_histories[track_id]