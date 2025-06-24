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
# Math modülünü içe aktar
import math
# SciPy signal processing için
from scipy import signal
# Time modülünü içe aktar
import time

# Logger oluştur
logger = logging.getLogger(__name__)


class AerialObjectClassifier:
    """Uçan nesneleri sınıflandır: Drone, Uçak, Helikopter, Kuş"""
    
    def __init__(self):
        """Gelişmiş hava aracı sınıflandırıcı"""
        # Nesne geçmişleri
        self.object_histories = {}
        self.history_length = 60  # 2 saniyelik geçmiş (30 FPS)
        
        # Sınıf özellikleri
        self.class_features = {
            'drone': {
                'size_range': (20, 20000),  # piksel²
                'speed_range': (0, 100),  # piksel/frame
                'hover_capability': True,
                'altitude_preference': 'low_to_medium',
                'aspect_ratio_range': (0.5, 2.0),
                'motion_smoothness': 0.7,
                'size_stability': 0.85,
                'typical_frequency': (50, 500),  # Hz - pervane dönüşü
                'night_lights': True,  # LED ışıkları
                'edge_sharpness': 0.8,
                'symmetry_score': 0.8
            },
            'airplane': {
                'size_range': (5000, 100000),  # piksel²
                'speed_range': (20, 200),  # piksel/frame
                'hover_capability': False,
                'altitude_preference': 'high',
                'aspect_ratio_range': (2.0, 5.0),  # Uzun gövde
                'motion_smoothness': 0.95,  # Çok düzgün
                'size_stability': 0.95,
                'typical_frequency': None,
                'night_lights': True,  # Navigation ışıkları
                'edge_sharpness': 0.6,
                'symmetry_score': 0.9
            },
            'helicopter': {
                'size_range': (10000, 50000),  # piksel²
                'speed_range': (0, 150),  # piksel/frame
                'hover_capability': True,
                'altitude_preference': 'medium',
                'aspect_ratio_range': (1.5, 3.0),
                'motion_smoothness': 0.8,
                'size_stability': 0.9,
                'typical_frequency': (10, 30),  # Hz - rotor
                'night_lights': True,
                'edge_sharpness': 0.5,  # Rotor blur
                'symmetry_score': 0.7
            },
            'bird': {
                'size_range': (50, 5000),  # piksel²
                'speed_range': (5, 80),  # piksel/frame
                'hover_capability': False,  # Bazı kuşlar hariç
                'altitude_preference': 'variable',
                'aspect_ratio_range': (0.3, 3.0),
                'motion_smoothness': 0.4,  # Düzensiz
                'size_stability': 0.5,  # Kanat çırpma
                'typical_frequency': (2, 20),  # Hz - kanat
                'night_lights': False,
                'edge_sharpness': 0.3,  # Yumuşak kenarlar
                'symmetry_score': 0.5
            }
        }
        
        # Gece/gündüz tespiti
        self.is_night_time = False
        self.night_detection_threshold = 60  # Ortalama parlaklık
        
        # Drone tespit hassasiyeti
        self.drone_sensitivity = 0.95  # Çok yüksek hassasiyet
        
    def classify(self, frame: np.ndarray, detection: Dict, 
                 track_id: Optional[str] = None) -> Tuple[str, float, Dict]:
        """
        Nesneyi sınıflandır
        
        Args:
            frame: Görüntü frame'i
            detection: Tespit bilgileri
            track_id: Takip ID'si
            
        Returns:
            (class_name, confidence, features)
        """
        if track_id is None:
            track_id = str(id(detection))
            
        # Gece/gündüz tespiti
        self._detect_day_night(frame)
        
        # Geçmişi güncelle
        self._update_history(track_id, detection, frame)
        
        # Tüm özellikleri analiz et
        features = self._extract_all_features(track_id, frame, detection)
        
        # Her sınıf için skor hesapla
        scores = {}
        for class_name in self.class_features:
            scores[class_name] = self._calculate_class_score(class_name, features)
            
        # Drone hassasiyetini artır
        if 'drone' in scores:
            scores['drone'] *= self.drone_sensitivity
            
        # En yüksek skoru bul
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]
        
        # Gece modunda drone tespitini güçlendir
        if self.is_night_time and best_class == 'drone':
            confidence = min(confidence * 1.2, 1.0)
            
        # Debug log
        if confidence > 0.5:
            logger.debug(f"Classified as {best_class} with confidence {confidence:.2f}")
            logger.debug(f"Scores: {scores}")
            logger.debug(f"Features: {features}")
            
        return best_class, confidence, features
        
    def _detect_day_night(self, frame: np.ndarray):
        """Gece/gündüz durumunu tespit et"""
        # HSV'ye çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # V kanalının ortalaması
        avg_brightness = np.mean(hsv[:, :, 2])
        
        # Gece tespiti
        self.is_night_time = avg_brightness < self.night_detection_threshold
        
    def _update_history(self, track_id: str, detection: Dict, frame: np.ndarray):
        """Nesne geçmişini güncelle"""
        if track_id not in self.object_histories:
            self.object_histories[track_id] = {
                'positions': deque(maxlen=self.history_length),
                'sizes': deque(maxlen=self.history_length),
                'aspects': deque(maxlen=self.history_length),
                'timestamps': deque(maxlen=self.history_length),
                'speeds': deque(maxlen=self.history_length),
                'brightness': deque(maxlen=self.history_length),
                'shapes': deque(maxlen=10),
                'edge_scores': deque(maxlen=self.history_length),
                'blur_scores': deque(maxlen=self.history_length)
            }
            
        history = self.object_histories[track_id]
        
        # Temel özellikler
        history['positions'].append(detection['center'])
        history['sizes'].append(detection['area'])
        history['aspects'].append(detection.get('aspect_ratio', 1.0))
        history['timestamps'].append(time.time())
        
        # Hız hesapla
        if len(history['positions']) > 1:
            prev_pos = history['positions'][-2]
            curr_pos = history['positions'][-1]
            speed = math.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                            (curr_pos[1] - prev_pos[1])**2)
            history['speeds'].append(speed)
            
        # ROI analizi
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # Parlaklık
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                history['brightness'].append(np.mean(gray_roi))
                
                # Edge skoru
                edges = cv2.Canny(gray_roi, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                history['edge_scores'].append(edge_density)
                
                # Blur skoru (motion blur veya rotor blur)
                laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
                blur_score = np.var(laplacian)
                history['blur_scores'].append(blur_score)
                
                # Şekil örneği
                if len(history['positions']) % 5 == 0:
                    roi_small = cv2.resize(roi, (64, 64))
                    history['shapes'].append(roi_small)
                    
    def _extract_all_features(self, track_id: str, frame: np.ndarray, 
                             detection: Dict) -> Dict:
        """Tüm özellikleri çıkar"""
        features = {}
        
        if track_id not in self.object_histories:
            return features
            
        history = self.object_histories[track_id]
        
        # Boyut özellikleri
        if history['sizes']:
            features['avg_size'] = np.mean(history['sizes'])
            features['size_variance'] = np.var(history['sizes']) / (features['avg_size'] + 1e-6)
            features['size_stability'] = 1.0 / (1.0 + features['size_variance'])
            
        # Hız özellikleri
        if history['speeds']:
            features['avg_speed'] = np.mean(history['speeds'])
            features['speed_variance'] = np.var(history['speeds'])
            features['motion_smoothness'] = 1.0 / (1.0 + features['speed_variance'])
            features['max_speed'] = np.max(history['speeds'])
            features['min_speed'] = np.min(history['speeds'])
            
        # Hover tespiti
        features.update(self._detect_hovering(history))
        
        # Yörünge analizi
        features.update(self._analyze_trajectory(history))
        
        # Frekans analizi
        features.update(self._analyze_frequency(history))
        
        # Görsel özellikler
        features.update(self._analyze_visual_features(frame, detection, history))
        
        # Işık tespiti (gece)
        if self.is_night_time:
            features.update(self._detect_lights(frame, detection))
            
        # En-boy oranı
        if history['aspects']:
            features['avg_aspect_ratio'] = np.mean(history['aspects'])
            features['aspect_variance'] = np.var(history['aspects'])
            
        # Edge ve blur özellikleri
        if history['edge_scores']:
            features['edge_sharpness'] = np.mean(history['edge_scores'])
            
        if history['blur_scores']:
            features['blur_score'] = np.mean(history['blur_scores'])
            features['has_motion_blur'] = features['blur_score'] < 100
            
        return features
        
    def _detect_hovering(self, history: Dict) -> Dict:
        """Hover tespiti"""
        features = {}
        
        if len(history['positions']) < 10:
            return features
            
        # Son 30 frame'i al
        recent_positions = list(history['positions'])[-30:]
        
        # Ortalama pozisyon
        mean_x = np.mean([p[0] for p in recent_positions])
        mean_y = np.mean([p[1] for p in recent_positions])
        
        # Sapma hesapla
        deviations = []
        for px, py in recent_positions:
            dist = math.sqrt((px - mean_x)**2 + (py - mean_y)**2)
            deviations.append(dist)
            
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        
        # Hover skoru
        hover_threshold = 10.0  # piksel
        features['is_hovering'] = max_deviation < hover_threshold
        features['hover_stability'] = 1.0 / (1.0 + avg_deviation)
        features['position_variance'] = avg_deviation
        
        return features
        
    def _analyze_trajectory(self, history: Dict) -> Dict:
        """Yörünge analizi"""
        features = {}
        
        positions = list(history['positions'])
        if len(positions) < 20:
            return features
            
        # Numpy array'e çevir
        pos_array = np.array(positions)
        
        # Doğrusallık analizi
        if len(positions) > 30:
            # Lineer regresyon
            x = pos_array[:, 0]
            y = pos_array[:, 1]
            
            # Polyfit
            coeffs = np.polyfit(x, y, 1)
            y_fit = np.polyval(coeffs, x)
            
            # R-squared
            ss_res = np.sum((y - y_fit)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-6))
            
            features['trajectory_linearity'] = r_squared
            features['is_straight_line'] = r_squared > 0.9
            
        # Yön değişimleri
        if len(history['speeds']) > 10:
            direction_changes = 0
            prev_dx, prev_dy = 0, 0
            
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                
                if i > 1:
                    # Yön değişimi kontrolü
                    dot_product = dx * prev_dx + dy * prev_dy
                    if dot_product < 0:  # 90 dereceden fazla dönüş
                        direction_changes += 1
                        
                prev_dx, prev_dy = dx, dy
                
            features['direction_changes'] = direction_changes
            features['maneuverability'] = direction_changes / len(positions)
            
        return features
        
    def _analyze_frequency(self, history: Dict) -> Dict:
        """Frekans analizi (pervane/rotor/kanat)"""
        features = {}
        
        # Boyut değişimi frekansı
        sizes = list(history['sizes'])
        if len(sizes) > 30:
            # Normalize et
            sizes_norm = (sizes - np.mean(sizes)) / (np.std(sizes) + 1e-6)
            
            # FFT
            fft = np.fft.fft(sizes_norm)
            freqs = np.fft.fftfreq(len(sizes_norm), d=1/30.0)  # 30 FPS
            
            # Pozitif frekanslar
            positive_freqs = freqs[:len(freqs)//2]
            fft_magnitude = np.abs(fft[:len(fft)//2])
            
            if len(positive_freqs) > 0:
                # Dominant frekans
                peak_idx = np.argmax(fft_magnitude)
                dominant_freq = positive_freqs[peak_idx]
                features['dominant_frequency'] = dominant_freq
                
                # Frekans bantları
                # Kuş kanat çırpması: 2-20 Hz
                bird_band = (positive_freqs >= 2) & (positive_freqs <= 20)
                # Helikopter rotor: 10-30 Hz
                heli_band = (positive_freqs >= 10) & (positive_freqs <= 30)
                # Drone pervane: 50-500 Hz (genelde görünmez ama titreşim)
                drone_band = (positive_freqs >= 50) & (positive_freqs <= 500)
                
                if np.any(bird_band):
                    features['bird_freq_power'] = np.max(fft_magnitude[bird_band])
                if np.any(heli_band):
                    features['heli_freq_power'] = np.max(fft_magnitude[heli_band])
                if np.any(drone_band):
                    features['drone_freq_power'] = np.max(fft_magnitude[drone_band])
                    
        return features
        
    def _analyze_visual_features(self, frame: np.ndarray, detection: Dict, 
                                history: Dict) -> Dict:
        """Görsel özellik analizi"""
        features = {}
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # ROI al
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return features
            
        # Gri tonlama
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Simetri analizi
        features.update(self._analyze_symmetry(gray))
        
        # Köşe tespiti (drone'lar keskin köşelere sahip)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                         qualityLevel=0.01, minDistance=5)
        if corners is not None:
            features['corner_count'] = len(corners)
            features['corner_density'] = len(corners) / (roi.size / 1000)
        else:
            features['corner_count'] = 0
            features['corner_density'] = 0
            
        # Kontur analizi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Konveks hull
            hull = cv2.convexHull(largest_contour)
            
            # Solidity
            contour_area = cv2.contourArea(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                features['solidity'] = contour_area / hull_area
            else:
                features['solidity'] = 0
                
            # Çevre/alan oranı
            perimeter = cv2.arcLength(largest_contour, True)
            if contour_area > 0:
                features['perimeter_ratio'] = perimeter / math.sqrt(contour_area)
            else:
                features['perimeter_ratio'] = 0
                
        # Texture analizi (drone'lar genelde düz texture)
        features['texture_variance'] = np.var(gray)
        
        return features
        
    def _analyze_symmetry(self, gray: np.ndarray) -> Dict:
        """Simetri analizi"""
        features = {}
        
        h, w = gray.shape
        
        # Yatay simetri
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        if right_half.shape[1] > 0:
            right_flipped = cv2.flip(right_half, 1)
            
            # Boyutları eşitle
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            if min_width > 0:
                left_half = left_half[:, :min_width]
                right_flipped = right_flipped[:, :min_width]
                
                # Benzerlik
                diff = cv2.absdiff(left_half, right_flipped)
                features['horizontal_symmetry'] = 1.0 - (np.mean(diff) / 255.0)
                
        # Dikey simetri
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        if bottom_half.shape[0] > 0:
            bottom_flipped = cv2.flip(bottom_half, 0)
            
            # Boyutları eşitle
            min_height = min(top_half.shape[0], bottom_flipped.shape[0])
            if min_height > 0:
                top_half = top_half[:min_height, :]
                bottom_flipped = bottom_flipped[:min_height, :]
                
                # Benzerlik
                diff = cv2.absdiff(top_half, bottom_flipped)
                features['vertical_symmetry'] = 1.0 - (np.mean(diff) / 255.0)
                
        # Ortalama simetri
        if 'horizontal_symmetry' in features and 'vertical_symmetry' in features:
            features['symmetry_score'] = (features['horizontal_symmetry'] + 
                                         features['vertical_symmetry']) / 2
        else:
            features['symmetry_score'] = 0
            
        return features
        
    def _detect_lights(self, frame: np.ndarray, detection: Dict) -> Dict:
        """Gece ışık tespiti"""
        features = {}
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # ROI al
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return features
            
        # HSV'ye çevir
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Yüksek parlaklık alanları
        bright_mask = hsv[:, :, 2] > 200
        bright_pixels = np.sum(bright_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        features['bright_pixel_ratio'] = bright_pixels / (total_pixels + 1e-6)
        features['has_lights'] = features['bright_pixel_ratio'] > 0.1
        
        # Işık renk analizi
        if bright_pixels > 10:
            bright_areas = roi[bright_mask]
            
            # Ortalama renk
            avg_color = np.mean(bright_areas, axis=0)
            
            # Beyaz ışık kontrolü (navigation lights)
            white_score = 1.0 - np.std(avg_color) / 128.0
            features['has_white_lights'] = white_score > 0.7
            
            # Renkli ışık kontrolü (red/green navigation lights)
            red_score = avg_color[2] / (np.sum(avg_color) + 1e-6)
            green_score = avg_color[1] / (np.sum(avg_color) + 1e-6)
            features['has_colored_lights'] = max(red_score, green_score) > 0.5
            
        return features
        
    def _calculate_class_score(self, class_name: str, features: Dict) -> float:
        """Belirli bir sınıf için skor hesapla"""
        class_info = self.class_features[class_name]
        score = 0.0
        weight_sum = 0.0
        
        # Boyut uyumu
        if 'avg_size' in features:
            size_min, size_max = class_info['size_range']
            if size_min <= features['avg_size'] <= size_max:
                size_score = 1.0
            else:
                # Uzaklığa göre azalan skor
                if features['avg_size'] < size_min:
                    size_score = features['avg_size'] / size_min
                else:
                    size_score = size_max / features['avg_size']
            score += size_score * 0.20
            weight_sum += 0.20
            
        # Hız uyumu
        if 'avg_speed' in features:
            speed_min, speed_max = class_info['speed_range']
            if speed_min <= features['avg_speed'] <= speed_max:
                speed_score = 1.0
            else:
                speed_score = 0.5
            score += speed_score * 0.15
            weight_sum += 0.15
            
        # Hover yeteneği
        if 'is_hovering' in features:
            if class_info['hover_capability']:
                hover_score = features['hover_stability'] if features['is_hovering'] else 0.3
            else:
                hover_score = 1.0 - features['hover_stability']
            score += hover_score * 0.15
            weight_sum += 0.15
            
        # Hareket düzgünlüğü
        if 'motion_smoothness' in features:
            target_smoothness = class_info['motion_smoothness']
            smoothness_diff = abs(features['motion_smoothness'] - target_smoothness)
            smoothness_score = 1.0 - smoothness_diff
            score += smoothness_score * 0.10
            weight_sum += 0.10
            
        # Boyut kararlılığı
        if 'size_stability' in features:
            target_stability = class_info['size_stability']
            stability_diff = abs(features['size_stability'] - target_stability)
            stability_score = 1.0 - stability_diff
            score += stability_score * 0.10
            weight_sum += 0.10
            
        # Simetri skoru
        if 'symmetry_score' in features:
            target_symmetry = class_info['symmetry_score']
            symmetry_diff = abs(features['symmetry_score'] - target_symmetry)
            symmetry_score = 1.0 - symmetry_diff
            score += symmetry_score * 0.10
            weight_sum += 0.10
            
        # Edge keskinliği
        if 'edge_sharpness' in features:
            target_sharpness = class_info['edge_sharpness']
            sharpness_diff = abs(features['edge_sharpness'] - target_sharpness)
            sharpness_score = 1.0 - sharpness_diff
            score += sharpness_score * 0.05
            weight_sum += 0.05
            
        # Gece ışıkları
        if self.is_night_time and 'has_lights' in features:
            if class_info['night_lights']:
                light_score = 1.0 if features['has_lights'] else 0.3
            else:
                light_score = 0.3 if features['has_lights'] else 1.0
            score += light_score * 0.10
            weight_sum += 0.10
            
        # Sınıfa özel özellikler
        if class_name == 'drone':
            # Drone-specific
            if 'corner_density' in features:
                # Drone'lar keskin köşelere sahip
                corner_score = min(features['corner_density'] / 5.0, 1.0)
                score += corner_score * 0.05
                weight_sum += 0.05
                
        elif class_name == 'airplane':
            # Airplane-specific
            if 'trajectory_linearity' in features:
                # Uçaklar düz hat boyunca uçar
                linear_score = features['trajectory_linearity']
                score += linear_score * 0.10
                weight_sum += 0.10
                
        elif class_name == 'helicopter':
            # Helicopter-specific
            if 'has_motion_blur' in features:
                # Helikopter rotoru blur yaratır
                blur_score = 1.0 if features['has_motion_blur'] else 0.5
                score += blur_score * 0.05
                weight_sum += 0.05
                
        elif class_name == 'bird':
            # Bird-specific
            if 'bird_freq_power' in features:
                # Kanat çırpma frekansı
                freq_score = min(features['bird_freq_power'] / 0.5, 1.0)
                score += freq_score * 0.10
                weight_sum += 0.10
                
        # Normalize et
        if weight_sum > 0:
            score = score / weight_sum
            
        return score
        
    def cleanup_old_tracks(self, active_track_ids: List[str]):
        """Eski takipleri temizle"""
        to_remove = []
        for track_id in self.object_histories:
            if track_id not in active_track_ids:
                to_remove.append(track_id)
                
        for track_id in to_remove:
            del self.object_histories[track_id]
            
    def get_night_mode_params(self) -> Dict:
        """Gece modu parametrelerini döndür"""
        return {
            'is_night': self.is_night_time,
            'enhancement_needed': self.is_night_time,
            'contrast_boost': 1.5 if self.is_night_time else 1.0,
            'brightness_boost': 20 if self.is_night_time else 0,
            'denoise': self.is_night_time
        }