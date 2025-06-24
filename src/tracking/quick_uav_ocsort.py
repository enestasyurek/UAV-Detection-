# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# SciPy'den doğrusal atama için optimize modülünü içe aktar
from scipy.optimize import linear_sum_assignment
# Kalman filtresi için filterpy kütüphanesinden içe aktar
from filterpy.kalman import KalmanFilter
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# OC-SORT temel sınıfını içe aktar
from .ocsort import OCSortTracker, OCSortTrack
# Loglama için logging modülünü içe aktar
import logging
# Matematiksel işlemler için math modülünü içe aktar
import math

# Logger oluştur
logger = logging.getLogger(__name__)


class QuickUAVTrack(OCSortTrack):
    """Quick-UAV için optimize edilmiş takip sınıfı"""
    
    def __init__(self, track_id: str, initial_detection: Dict):
        """
        Quick-UAV takip nesnesi
        
        Args:
            track_id: Takip ID'si
            initial_detection: İlk tespit
        """
        # Temel sınıfı başlat
        super().__init__(track_id, initial_detection)
        
        # UAV-specific özellikler
        self.altitude_estimate = 0  # Tahmini irtifa
        self.speed_estimate = 0  # Tahmini hız (m/s)
        self.direction_angle = 0  # Hareket yönü (derece)
        self.is_hovering = False  # Havada asılı kalma durumu
        
        # Hareket pattern analizi için
        self.motion_pattern = "unknown"  # linear, circular, hovering, erratic
        self.pattern_confidence = 0.0
        
        # UAV boyut tahmini (küçük, orta, büyük drone)
        self.size_category = "medium"
        
        # Gelişmiş hız tahmini için
        self.acceleration_history = []
        self.jerk_history = []  # İvme değişimi (jerk)
        
    def update_motion_analysis(self):
        """UAV hareket pattern analizi"""
        if len(self.center_history) < 5:
            return
            
        # Son 5 pozisyonu al
        recent_positions = self.center_history[-5:]
        
        # Hız ve ivme hesapla
        velocities = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            speed = math.sqrt(dx**2 + dy**2)
            velocities.append(speed)
            
        # Ortalama hız
        avg_speed = np.mean(velocities)
        self.speed_estimate = avg_speed
        
        # Hovering tespiti
        if avg_speed < 2.0:  # piksel/frame
            self.is_hovering = True
            self.motion_pattern = "hovering"
            self.pattern_confidence = 0.9
            return
            
        # Yön değişimi analizi
        angles = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            angle = math.atan2(dy, dx) * 180 / math.pi
            angles.append(angle)
            
        # Yön varyansı
        angle_variance = np.var(angles) if angles else 0
        
        # Pattern belirleme
        if angle_variance < 10:  # Düşük varyans = doğrusal hareket
            self.motion_pattern = "linear"
            self.pattern_confidence = 0.8
            self.direction_angle = np.mean(angles)
        elif angle_variance > 100:  # Yüksek varyans = düzensiz hareket
            self.motion_pattern = "erratic"
            self.pattern_confidence = 0.7
        else:  # Orta varyans = dairesel hareket olabilir
            self.motion_pattern = "circular"
            self.pattern_confidence = 0.6
            
    def estimate_altitude(self, bbox_area: float, reference_area: float = 5000):
        """
        Bbox alanına göre tahmini irtifa hesapla
        
        Args:
            bbox_area: Mevcut bbox alanı
            reference_area: Referans alan (örn: 10m irtifada)
        """
        # Basit ters orantı modeli: irtifa arttıkça alan küçülür
        if bbox_area > 0:
            # Tahmini irtifa (metre)
            self.altitude_estimate = 10 * math.sqrt(reference_area / bbox_area)
            
    def predict_future_position(self, frames_ahead: int = 5) -> Tuple[int, int]:
        """
        Gelecekteki pozisyonu tahmin et (UAV hareketi için optimize)
        
        Args:
            frames_ahead: Kaç frame sonrası için tahmin
            
        Returns:
            Tahmin edilen pozisyon
        """
        if self.is_hovering:
            # Hovering durumunda küçük rastgele hareketler
            current = self.center_history[-1]
            noise_x = np.random.normal(0, 1)
            noise_y = np.random.normal(0, 1)
            return (int(current[0] + noise_x), int(current[1] + noise_y))
            
        # Kalman tahmini kullan
        future_state = self.kf.x.copy()
        for _ in range(frames_ahead):
            future_state = self.kf.F @ future_state
            
        return (int(future_state[0]), int(future_state[1]))


class QuickUAVOCSortTracker(OCSortTracker):
    """
    Quick-UAV OC-SORT: Drone takibi için optimize edilmiş OC-SORT
    """
    
    def __init__(self,
                 det_thresh: float = 0.3,  # Drone'lar için daha düşük eşik
                 max_age: int = 50,  # Drone'lar için daha uzun takip
                 min_hits: int = 2,  # Daha hızlı onay
                 iou_threshold: float = 0.2,  # Daha esnek eşleştirme
                 delta_t: int = 3,
                 use_byte: bool = True,
                 altitude_aware: bool = True,
                 motion_prediction: bool = True):
        """
        Quick-UAV OC-SORT başlatıcı
        
        Args:
            det_thresh: Tespit eşiği
            max_age: Maksimum kayıp frame
            min_hits: Minimum onay için görülme
            iou_threshold: IoU eşiği
            delta_t: Hız hesaplama frame aralığı
            use_byte: ByteTrack benzeri düşük skor kullanımı
            altitude_aware: İrtifa tahmini kullan
            motion_prediction: Gelişmiş hareket tahmini
        """
        # Temel sınıfı başlat
        super().__init__(det_thresh, max_age, min_hits, iou_threshold, delta_t, use_byte)
        
        # UAV-specific parametreler
        self.altitude_aware = altitude_aware
        self.motion_prediction = motion_prediction
        
        # Drone takibi için optimize edilmiş parametreler
        self.small_object_threshold = 500  # Küçük nesne alanı eşiği
        self.fast_motion_threshold = 50  # Hızlı hareket eşiği (piksel/frame)
        self.hover_threshold = 3  # Hovering tespiti için eşik
        
        # Adaptif parametreler
        self.adaptive_iou = True  # Hıza göre IoU eşiği ayarla
        self.adaptive_age = True  # Boyuta göre max_age ayarla
        
        logger.info("Quick-UAV OC-SORT initialized with drone-optimized parameters")
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Quick-UAV optimize edilmiş takip güncelleme
        
        Args:
            detections: YOLO tespitleri
            
        Returns:
            Takip ID'li tespitler
        """
        # Frame sayacını artır
        self.frame_count += 1
        
        # Drone-specific ön işleme
        detections = self._preprocess_drone_detections(detections)
        
        # Tahminleri yap (hareket pattern'e göre)
        for tracker in self.trackers:
            # Hareket analizi güncelle
            tracker.update_motion_analysis()
            
            # Pattern'e göre tahmin yap
            if self.motion_prediction:
                if tracker.motion_pattern == "hovering":
                    # Hovering için minimal tahmin
                    tracker.kf.F[0, 4] = 0.1  # Düşük hız katsayısı
                    tracker.kf.F[1, 5] = 0.1
                elif tracker.motion_pattern == "linear":
                    # Doğrusal hareket için standart tahmin
                    tracker.kf.F[0, 4] = 1.0
                    tracker.kf.F[1, 5] = 1.0
                elif tracker.motion_pattern == "erratic":
                    # Düzensiz hareket için yüksek belirsizlik
                    tracker.kf.Q *= 2.0
                    
            tracker.predict()
            
        # Tespitleri boyuta göre kategorize et
        small_dets = []
        normal_dets = []
        
        for det in detections:
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if area < self.small_object_threshold:
                small_dets.append(det)
            else:
                normal_dets.append(det)
                
        # Önce normal boyutlu drone'ları eşleştir
        all_detections = normal_dets + small_dets
        
        # Adaptif IoU eşiği hesapla
        if self.adaptive_iou:
            adaptive_thresholds = self._calculate_adaptive_thresholds()
        else:
            adaptive_thresholds = [self.iou_threshold] * len(self.trackers)
            
        # Temel OC-SORT eşleştirmesi (modifiye edilmiş)
        matched1, unmatched_trackers1, unmatched_dets1 = self._associate_adaptive(
            self.trackers, all_detections, adaptive_thresholds
        )
        
        # Eşleşen takipleri güncelle
        for t_idx, d_idx in matched1:
            tracker = self.trackers[t_idx]
            detection = all_detections[d_idx]
            
            tracker.update(detection)
            tracker.update_kalman(detection['bbox'])
            
            # İrtifa tahmini
            if self.altitude_aware:
                bbox = detection['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                tracker.estimate_altitude(area)
                
        # Küçük nesneler için özel işlem
        if unmatched_trackers1 and small_dets:
            # Küçük nesneler için daha esnek eşleştirme
            remaining_trackers = [self.trackers[i] for i in unmatched_trackers1]
            remaining_small_dets = [d for d in small_dets if d in [all_detections[i] for i in unmatched_dets1]]
            
            if remaining_small_dets:
                matched2, unmatched_trackers2, unmatched_dets2 = self._associate_small_objects(
                    remaining_trackers, remaining_small_dets
                )
                
                # Güncelle
                for t_idx, d_idx in matched2:
                    actual_t_idx = unmatched_trackers1[t_idx]
                    self.trackers[actual_t_idx].update(remaining_small_dets[d_idx])
                    self.trackers[actual_t_idx].update_kalman(remaining_small_dets[d_idx]['bbox'])
                    
        # Kayıp takipleri işle
        for idx in unmatched_trackers1:
            self.trackers[idx].mark_lost()
            
        # Yeni takipler oluştur
        for idx in unmatched_dets1:
            self._init_track(all_detections[idx])
            
        # Gelişmiş yeniden ilişkilendirme (motion prediction kullanarak)
        if self.motion_prediction:
            lost_trackers = [t for t in self.trackers if t.lost_frames > 0 and t.lost_frames < 10]
            if lost_trackers:
                self._reassociate_with_prediction(lost_trackers, 
                                                 [all_detections[i] for i in unmatched_dets1])
                
        # Adaptif yaş kontrolü
        if self.adaptive_age:
            self._adaptive_track_management()
        else:
            self.trackers = [t for t in self.trackers if t.lost_frames < self.max_age]
            
        # Sonuçları hazırla
        results = []
        for tracker in self.trackers:
            if tracker.age >= self.min_hits and tracker.lost_frames == 0:
                bbox = tracker.get_latest_bbox()
                detection = {
                    'bbox': bbox,
                    'center': tracker.get_latest_center(),
                    'confidence': tracker.confidence_history[-1],
                    'class': tracker.class_name,
                    'track_id': tracker.track_id,
                    'age': tracker.age,
                    'motion_pattern': tracker.motion_pattern,
                    'pattern_confidence': tracker.pattern_confidence,
                    'is_hovering': tracker.is_hovering,
                    'speed_estimate': tracker.speed_estimate,
                    'altitude_estimate': tracker.altitude_estimate,
                    'direction_angle': tracker.direction_angle
                }
                results.append(detection)
                
        return results
        
    def _preprocess_drone_detections(self, detections: List[Dict]) -> List[Dict]:
        """Drone tespitleri için ön işleme"""
        processed = []
        
        for det in detections:
            # Drone sınıfı kontrolü ve güven skoru ayarı
            if det['class'] in ['drone', 'uav', 'quadcopter', 'airplane', 'bird']:
                # Drone benzeri nesneler için güven skorunu artır
                if det['class'] in ['drone', 'uav', 'quadcopter']:
                    det['confidence'] = min(det['confidence'] * 1.1, 1.0)
                    
                # Küçük nesneler için ek işlem
                bbox = det['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                if area < self.small_object_threshold:
                    # Küçük nesneler için güven eşiğini düşür
                    if det['confidence'] >= self.det_thresh * 0.7:
                        processed.append(det)
                else:
                    if det['confidence'] >= self.det_thresh:
                        processed.append(det)
            else:
                # Diğer nesneler için normal işlem
                if det['confidence'] >= self.det_thresh:
                    processed.append(det)
                    
        return processed
        
    def _calculate_adaptive_thresholds(self) -> List[float]:
        """Her takip için adaptif IoU eşiği hesapla"""
        thresholds = []
        
        for tracker in self.trackers:
            base_threshold = self.iou_threshold
            
            # Hıza göre ayarla
            if tracker.speed_estimate > self.fast_motion_threshold:
                # Hızlı hareket için daha esnek eşik
                speed_factor = min(tracker.speed_estimate / self.fast_motion_threshold, 2.0)
                base_threshold *= (1.0 / speed_factor)
                
            # Boyuta göre ayarla
            if hasattr(tracker, 'size_category'):
                if tracker.size_category == "small":
                    base_threshold *= 0.7  # Küçük drone'lar için daha esnek
                elif tracker.size_category == "large":
                    base_threshold *= 1.2  # Büyük drone'lar için daha sıkı
                    
            # Hovering durumu
            if tracker.is_hovering:
                base_threshold *= 1.5  # Hovering için daha sıkı eşleştirme
                
            # Pattern güvenilirliğine göre
            if tracker.pattern_confidence > 0.7:
                base_threshold *= 0.9  # Güvenilir pattern için hafif esneklik
                
            thresholds.append(max(0.1, min(0.9, base_threshold)))
            
        return thresholds
        
    def _associate_adaptive(self, trackers: List[QuickUAVTrack],
                           detections: List[Dict],
                           adaptive_thresholds: List[float]) -> Tuple:
        """Adaptif eşiklerle ilişkilendirme"""
        if not trackers or not detections:
            return [], list(range(len(trackers))), list(range(len(detections)))
            
        # IoU matrisi + hareket tahmini
        cost_matrix = np.zeros((len(trackers), len(detections)))
        
        for t_idx, tracker in enumerate(trackers):
            # Tahmin edilen pozisyon
            if self.motion_prediction and tracker.pattern_confidence > 0.5:
                pred_center = tracker.predict_future_position(2)
                # Tahmin edilen bbox
                last_bbox = tracker.get_latest_bbox()
                w = last_bbox[2] - last_bbox[0]
                h = last_bbox[3] - last_bbox[1]
                pred_bbox = [
                    pred_center[0] - w//2, pred_center[1] - h//2,
                    pred_center[0] + w//2, pred_center[1] + h//2
                ]
            else:
                pred_bbox = self._state_to_bbox(tracker.kf.x)
                
            for d_idx, det in enumerate(detections):
                # IoU hesapla
                iou = self._calculate_iou(pred_bbox, det['bbox'])
                
                # Mesafe bazlı maliyet
                pred_center = ((pred_bbox[0] + pred_bbox[2]) / 2, 
                              (pred_bbox[1] + pred_bbox[3]) / 2)
                det_center = det['center']
                distance = math.sqrt((pred_center[0] - det_center[0])**2 + 
                                   (pred_center[1] - det_center[1])**2)
                
                # Kombine maliyet
                if tracker.is_hovering:
                    # Hovering için mesafe daha önemli
                    cost = 0.3 * (1 - iou) + 0.7 * (distance / 100)
                else:
                    # Normal hareket için IoU daha önemli
                    cost = 0.7 * (1 - iou) + 0.3 * (distance / 100)
                    
                cost_matrix[t_idx, d_idx] = cost
                
        # Hungarian algoritması
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Adaptif eşik kontrolü
        matched = []
        for r, c in zip(row_ind, col_ind):
            # Maliyet eşiği (1 - adaptive_threshold)
            if cost_matrix[r, c] <= (1 - adaptive_thresholds[r]):
                matched.append((r, c))
                
        # Eşleşmeyenler
        unmatched_trackers = list(set(range(len(trackers))) - set([m[0] for m in matched]))
        unmatched_dets = list(set(range(len(detections))) - set([m[1] for m in matched]))
        
        return matched, unmatched_trackers, unmatched_dets
        
    def _associate_small_objects(self, trackers: List[QuickUAVTrack],
                                detections: List[Dict]) -> Tuple:
        """Küçük nesneler için özel eşleştirme"""
        if not trackers or not detections:
            return [], list(range(len(trackers))), list(range(len(detections)))
            
        # Mesafe tabanlı eşleştirme (küçük nesneler için)
        cost_matrix = np.zeros((len(trackers), len(detections)))
        
        for t_idx, tracker in enumerate(trackers):
            pred_center = tracker.predict_future_position(1)
            
            for d_idx, det in enumerate(detections):
                # Sadece mesafe bazlı
                distance = math.sqrt((pred_center[0] - det['center'][0])**2 + 
                                   (pred_center[1] - det['center'][1])**2)
                                   
                # Hıza göre normalize et
                normalized_distance = distance / (1 + tracker.speed_estimate * 0.1)
                cost_matrix[t_idx, d_idx] = normalized_distance
                
        # Eşleştir
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Mesafe eşiği (piksel)
        distance_threshold = 100  # Küçük nesneler için geniş eşik
        
        matched = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= distance_threshold:
                matched.append((r, c))
                
        unmatched_trackers = list(set(range(len(trackers))) - set([m[0] for m in matched]))
        unmatched_dets = list(set(range(len(detections))) - set([m[1] for m in matched]))
        
        return matched, unmatched_trackers, unmatched_dets
        
    def _reassociate_with_prediction(self, lost_trackers: List[QuickUAVTrack],
                                    remaining_dets: List[Dict]):
        """Hareket tahmini ile yeniden ilişkilendirme"""
        if not lost_trackers or not remaining_dets:
            return
            
        associations = []
        
        for tracker in lost_trackers:
            if tracker.pattern_confidence < 0.5:
                continue
                
            # Gelecek pozisyon tahmini
            future_frames = min(tracker.lost_frames + 2, 5)
            pred_center = tracker.predict_future_position(future_frames)
            
            best_match = None
            best_distance = float('inf')
            
            for det in remaining_dets:
                distance = math.sqrt((pred_center[0] - det['center'][0])**2 + 
                                   (pred_center[1] - det['center'][1])**2)
                                   
                # Pattern'e göre eşik belirle
                if tracker.motion_pattern == "linear":
                    threshold = 50 + 10 * future_frames
                elif tracker.motion_pattern == "hovering":
                    threshold = 30
                else:
                    threshold = 70 + 15 * future_frames
                    
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match = det
                    
            if best_match:
                associations.append((tracker, best_match))
                remaining_dets.remove(best_match)
                
        # İlişkilendirmeleri uygula
        for tracker, det in associations:
            tracker.update(det)
            tracker.update_kalman(det['bbox'])
            tracker.lost_frames = 0
            
    def _adaptive_track_management(self):
        """Adaptif takip yönetimi"""
        updated_trackers = []
        
        for tracker in self.trackers:
            # Boyuta göre max_age ayarla
            if hasattr(tracker, 'size_category'):
                if tracker.size_category == "small":
                    # Küçük drone'lar için daha uzun takip
                    max_age = self.max_age * 1.5
                elif tracker.size_category == "large":
                    # Büyük drone'lar için standart
                    max_age = self.max_age
                else:
                    max_age = self.max_age
            else:
                max_age = self.max_age
                
            # Hovering durumu için özel işlem
            if tracker.is_hovering and tracker.lost_frames > 0:
                # Hovering drone'ları daha uzun süre takip et
                max_age *= 2
                
            # Pattern güvenilirliğine göre
            if tracker.pattern_confidence > 0.7:
                max_age *= 1.2
                
            if tracker.lost_frames < max_age:
                updated_trackers.append(tracker)
                
        self.trackers = updated_trackers
        
    def _init_track(self, detection: Dict):
        """Yeni UAV takibi başlat"""
        track_id = self._generate_track_id()
        tracker = QuickUAVTrack(track_id, detection)
        
        # Boyut kategorisi belirle
        bbox = detection['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if area < 1000:
            tracker.size_category = "small"
        elif area > 10000:
            tracker.size_category = "large"
        else:
            tracker.size_category = "medium"
            
        self.trackers.append(tracker)
        
    def get_tracking_analytics(self) -> Dict:
        """Detaylı takip analitiği"""
        analytics = {
            'total_tracks': len(self.trackers),
            'active_tracks': len([t for t in self.trackers if t.lost_frames == 0]),
            'hovering_drones': len([t for t in self.trackers if t.is_hovering]),
            'motion_patterns': {},
            'size_distribution': {},
            'average_speed': 0,
            'average_altitude': 0
        }
        
        # Motion pattern dağılımı
        patterns = [t.motion_pattern for t in self.trackers if t.lost_frames == 0]
        for pattern in set(patterns):
            analytics['motion_patterns'][pattern] = patterns.count(pattern)
            
        # Boyut dağılımı
        sizes = [t.size_category for t in self.trackers if hasattr(t, 'size_category')]
        for size in set(sizes):
            analytics['size_distribution'][size] = sizes.count(size)
            
        # Ortalama hız ve irtifa
        active_trackers = [t for t in self.trackers if t.lost_frames == 0]
        if active_trackers:
            speeds = [t.speed_estimate for t in active_trackers]
            altitudes = [t.altitude_estimate for t in active_trackers]
            analytics['average_speed'] = np.mean(speeds)
            analytics['average_altitude'] = np.mean(altitudes)
            
        return analytics