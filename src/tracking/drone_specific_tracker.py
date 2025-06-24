# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# SciPy'den doğrusal atama için optimize modülünü içe aktar
from scipy.optimize import linear_sum_assignment
# Kalman filtresi için filterpy kütüphanesinden içe aktar
from filterpy.kalman import KalmanFilter
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# QuickUAV tracker'ı temel al
from .quick_uav_ocsort import QuickUAVOCSortTracker, QuickUAVTrack
# Loglama için logging modülünü içe aktar
import logging
# Matematiksel işlemler için math modülünü içe aktar
import math
# OpenCV'yi görüntü işleme için içe aktar
import cv2

# Logger oluştur
logger = logging.getLogger(__name__)


class DroneTrack(QuickUAVTrack):
    """Drone'lar için özelleştirilmiş takip sınıfı"""
    
    def __init__(self, track_id: str, initial_detection: Dict):
        """
        Drone takip nesnesi
        
        Args:
            track_id: Takip ID'si
            initial_detection: İlk tespit
        """
        # Temel sınıfı başlat
        super().__init__(track_id, initial_detection)
        
        # Drone-specific özellikler
        self.range_category = self._determine_range(initial_detection.get('area', 0))
        self.stability_score = 1.0  # Takip kararlılığı
        self.visual_similarity_history = []  # Görsel benzerlik geçmişi
        self.predicted_trajectory = []  # Tahmin edilen yörünge
        
        # Mesafe tahmini - Kalman'dan önce başlat
        self.estimated_distance = self._estimate_distance(initial_detection.get('area', 1000))
        
        # Gelişmiş Kalman filtresi (9 durum: x, y, z, s, r, dx, dy, dz, ds)
        self.kf = KalmanFilter(dim_x=9, dim_z=4)
        self._init_advanced_kalman(initial_detection['bbox'])
        
        # Takip güvenilirliği
        self.tracking_confidence = 1.0
        
        # Görsel özellik için
        self.last_visual_feature = None
        
        # Geçerli bbox sakla
        self.last_valid_bbox = initial_detection['bbox']
        
    def _determine_range(self, area: float) -> str:
        """Mesafe kategorisi belirle"""
        if area > 10000:
            return "near"
        elif area > 1000:
            return "medium"
        else:
            return "far"
            
    def _estimate_distance(self, area: float) -> float:
        """Alan bazlı mesafe tahmini (metre)"""
        # Basit ters kare kök modeli
        # Referans: 1000 piksel alan = 10 metre
        reference_area = 1000
        reference_distance = 10
        
        if area > 0:
            distance = reference_distance * math.sqrt(reference_area / area)
            return min(max(distance, 1), 500)  # 1-500m arası
        return 50  # Varsayılan
        
    def _bbox_to_measurement(self, bbox: List[int]) -> np.ndarray:
        """Bbox'ı Kalman ölçümüne çevir"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1) if y2 > y1 else 1.0
        return np.array([cx, cy, s, r])
        
    def _init_advanced_kalman(self, bbox: List[int]):
        """Gelişmiş Kalman filtresi başlatma"""
        # Durum: [x, y, z, s, r, dx, dy, dz, ds]
        # z: tahmini yükseklik, diğerleri standart
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1) if y2 > y1 else 1.0
        z = self.estimated_distance * 0.5  # Basit yükseklik tahmini
        
        # Başlangıç durumu
        self.kf.x = np.array([cx, cy, z, s, r, 0, 0, 0, 0])
        
        # Durum geçiş matrisi
        self.kf.F = np.eye(9)
        self.kf.F[0, 5] = 1  # x += dx
        self.kf.F[1, 6] = 1  # y += dy
        self.kf.F[2, 7] = 1  # z += dz
        self.kf.F[3, 8] = 1  # s += ds
        
        # Ölçüm matrisi (x, y, s, r ölçüyoruz)
        self.kf.H = np.zeros((4, 9))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 3] = 1  # s
        self.kf.H[3, 4] = 1  # r
        
        # Belirsizlikler
        self.kf.R *= 10.0
        self.kf.P *= 100.0
        self.kf.Q[5:, 5:] *= 0.01
        
    def update_with_confidence(self, detection: Dict, visual_similarity: float = 1.0):
        """Güven skorlu güncelleme"""
        # Temel güncelleme
        self.update(detection)
        
        # Kalman güncellemesi
        measurement = self._bbox_to_measurement(detection['bbox'])
        self.kf.update(measurement)
        
        # Görsel benzerlik geçmişi
        self.visual_similarity_history.append(visual_similarity)
        if len(self.visual_similarity_history) > 10:
            self.visual_similarity_history.pop(0)
            
        # Takip güvenilirliğini güncelle
        self._update_tracking_confidence()
        
        # Mesafe güncellemesi
        self.estimated_distance = self._estimate_distance(detection['area'])
        self.range_category = self._determine_range(detection['area'])
        
    def _update_tracking_confidence(self):
        """Takip güvenilirliğini hesapla"""
        # Faktörler:
        # 1. Görsel benzerlik tutarlılığı
        # 2. Hareket tutarlılığı
        # 3. Boyut tutarlılığı
        
        confidence = 1.0
        
        # Görsel benzerlik
        if self.visual_similarity_history:
            avg_similarity = np.mean(self.visual_similarity_history)
            confidence *= avg_similarity
            
        # Hareket tutarlılığı (hız varyansı)
        if len(self.velocity_history) > 2:
            velocity_var = np.var(self.velocity_history, axis=0).mean()
            motion_consistency = 1.0 / (1.0 + velocity_var / 100.0)
            confidence *= motion_consistency
            
        # Boyut tutarlılığı
        if len(self.bbox_history) > 2:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in self.bbox_history[-5:]]
            area_var = np.var(areas)
            size_consistency = 1.0 / (1.0 + area_var / 10000.0)
            confidence *= size_consistency
            
        self.tracking_confidence = confidence
        
    def predict_future_trajectory(self, frames_ahead: int = 30) -> List[Tuple[int, int]]:
        """Gelecek yörünge tahmini"""
        trajectory = []
        
        # Mevcut durumdan başla
        state = self.kf.x.copy()
        
        for _ in range(frames_ahead):
            # Bir adım tahmin
            state = self.kf.F @ state
            
            # Pozisyonu kaydet
            trajectory.append((int(state[0]), int(state[1])))
            
        self.predicted_trajectory = trajectory
        return trajectory


class DroneSpecificTracker(QuickUAVOCSortTracker):
    """Drone'lar için özel olarak optimize edilmiş takip algoritması"""
    
    def __init__(self,
                 det_thresh: float = 0.1,  # Ultra düşük
                 max_age: int = 100,  # Çok uzun takip
                 min_hits: int = 1,  # Hemen onay
                 iou_threshold: float = 0.1,  # Çok esnek
                 use_visual_features: bool = True,
                 range_adaptive: bool = True):
        """
        Drone-specific tracker
        
        Args:
            det_thresh: Tespit eşiği
            max_age: Maksimum kayıp frame
            min_hits: Minimum onay
            iou_threshold: IoU eşiği
            use_visual_features: Görsel özellik kullan
            range_adaptive: Mesafeye göre adaptif parametreler
        """
        # Temel sınıfı başlat
        super().__init__(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            altitude_aware=True,
            motion_prediction=True
        )
        
        # Drone-specific parametreler
        self.use_visual_features = use_visual_features
        self.range_adaptive = range_adaptive
        
        # Mesafeye göre adaptif parametreler
        self.range_params = {
            'near': {'iou': 0.3, 'max_age': 50, 'min_dist': 50},
            'medium': {'iou': 0.2, 'max_age': 75, 'min_dist': 100},
            'far': {'iou': 0.1, 'max_age': 100, 'min_dist': 200}
        }
        
        # Görsel özellik eşleştirme için histogram
        self.use_histogram = True
        
        logger.info("DroneSpecificTracker initialized")
        
    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Drone-optimized takip güncelleme
        
        Args:
            detections: Tespitler
            frame: Görüntü frame'i (görsel özellikler için)
            
        Returns:
            Takip edilmiş tespitler
        """
        # Frame sayacını artır
        self.frame_count += 1
        
        # Görsel özellikleri çıkar
        visual_features = {}
        if self.use_visual_features and frame is not None:
            for det in detections:
                feature = self._extract_visual_feature(frame, det['bbox'])
                visual_features[id(det)] = feature
                
        # Tahminleri yap
        for tracker in self.trackers:
            tracker.predict()
            
        # Mesafeye göre grupla
        if self.range_adaptive:
            range_groups = self._group_by_range(detections)
            all_matches = []
            
            # Her mesafe grubu için ayrı eşleştirme
            for range_cat, group_dets in range_groups.items():
                if not group_dets:
                    continue
                    
                # Bu mesafe için parametreleri al
                params = self.range_params[range_cat]
                
                # Bu mesafedeki tracker'ları bul
                range_trackers = [t for t in self.trackers 
                                 if hasattr(t, 'range_category') and t.range_category == range_cat]
                
                if range_trackers:
                    # Eşleştir
                    matches = self._match_with_visual_features(
                        range_trackers, group_dets, 
                        visual_features, params['iou']
                    )
                    all_matches.extend(matches)
                    
            # Geri kalan eşleştirmeler
            matched_det_ids = {id(match[1]) for match in all_matches}
            unmatched_dets = [d for d in detections 
                             if id(d) not in matched_det_ids]
            
            matched_tracker_ids = {id(match[0]) for match in all_matches}
            remaining_trackers = [t for t in self.trackers 
                                 if id(t) not in matched_tracker_ids]
            
            if remaining_trackers and unmatched_dets:
                cross_range_matches = self._match_with_visual_features(
                    remaining_trackers, unmatched_dets,
                    visual_features, self.iou_threshold * 0.5
                )
                all_matches.extend(cross_range_matches)
                
        else:
            # Standart eşleştirme
            all_matches = self._match_with_visual_features(
                self.trackers, detections, visual_features, self.iou_threshold
            )
            
        # Güncellemeleri uygula
        updated_trackers = set()
        for tracker, det in all_matches:
            visual_sim = self._calculate_visual_similarity(
                tracker, visual_features.get(id(det), None)
            )
            tracker.update_with_confidence(det, visual_sim)
            updated_trackers.add(tracker)
            
        # Eşleşmeyen tracker'ları kayıp işaretle
        for tracker in self.trackers:
            if tracker not in updated_trackers:
                tracker.mark_lost()
                
        # Yeni tracker'lar oluştur
        matched_det_ids = {id(det) for _, det in all_matches}
        for det in detections:
            if id(det) not in matched_det_ids:
                self._init_track(det)
                
        # Tracker yönetimi
        self._manage_tracks()
        
        # Sonuçları hazırla
        results = []
        for tracker in self.trackers:
            if tracker.age >= self.min_hits and tracker.lost_frames == 0:
                # Yörünge tahmini
                trajectory = tracker.predict_future_trajectory(10)
                
                result = {
                    'bbox': tracker.get_latest_bbox(),
                    'center': tracker.get_latest_center(),
                    'confidence': tracker.confidence_history[-1],
                    'class': 'drone',
                    'track_id': tracker.track_id,
                    'age': tracker.age,
                    'range': tracker.range_category,
                    'distance': tracker.estimated_distance,
                    'tracking_confidence': tracker.tracking_confidence,
                    'motion_pattern': tracker.motion_pattern,
                    'is_hovering': tracker.is_hovering,
                    'trajectory': trajectory[:5]  # İlk 5 tahmin
                }
                results.append(result)
                
        return results
        
    def _group_by_range(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Tespitleri mesafeye göre grupla"""
        groups = {'near': [], 'medium': [], 'far': []}
        
        for det in detections:
            area = det.get('area', 0)
            if area > 10000:
                groups['near'].append(det)
            elif area > 1000:
                groups['medium'].append(det)
            else:
                groups['far'].append(det)
                
        return groups
        
    def _extract_visual_feature(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Görsel özellik çıkarımı"""
        x1, y1, x2, y2 = bbox
        
        # Sınırları kontrol et
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Bölgeyi kırp
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(256)  # Boş özellik
            
        # Renk histogramı
        hist_b = cv2.calcHist([roi], [0], None, [64], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [64], [0, 256])
        hist_r = cv2.calcHist([roi], [2], None, [64], [0, 256])
        hist_gray = cv2.calcHist([cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)], 
                                [0], None, [64], [0, 256])
        
        # Birleştir ve normalize et
        feature = np.concatenate([hist_b, hist_g, hist_r, hist_gray]).flatten()
        feature = feature / (np.linalg.norm(feature) + 1e-6)
        
        return feature
        
    def _match_with_visual_features(self, trackers: List[DroneTrack],
                                   detections: List[Dict],
                                   visual_features: Dict,
                                   iou_thresh: float) -> List[Tuple]:
        """Görsel özellik destekli eşleştirme"""
        if not trackers or not detections:
            return []
            
        # Maliyet matrisi (IoU + görsel benzerlik)
        cost_matrix = np.zeros((len(trackers), len(detections)))
        
        for i, tracker in enumerate(trackers):
            pred_bbox = self._state_to_bbox(tracker.kf.x)
            
            for j, det in enumerate(detections):
                # IoU maliyeti
                iou = self._calculate_iou(pred_bbox, det['bbox'])
                iou_cost = 1 - iou
                
                # Görsel benzerlik maliyeti
                if self.use_visual_features and id(det) in visual_features:
                    visual_sim = self._calculate_visual_similarity(
                        tracker, visual_features[id(det)]
                    )
                    visual_cost = 1 - visual_sim
                else:
                    visual_cost = 0.5  # Nötr
                    
                # Mesafe maliyeti
                pred_center = ((pred_bbox[0] + pred_bbox[2]) / 2,
                              (pred_bbox[1] + pred_bbox[3]) / 2)
                det_center = det['center']
                distance = math.sqrt((pred_center[0] - det_center[0])**2 + 
                                   (pred_center[1] - det_center[1])**2)
                                   
                # Mesafeye göre normalize
                if hasattr(tracker, 'range_category'):
                    range_params = self.range_params.get(tracker.range_category, 
                                                        self.range_params['medium'])
                else:
                    range_params = self.range_params['medium']
                norm_distance = distance / range_params['min_dist']
                
                # Kombine maliyet
                cost = 0.4 * iou_cost + 0.3 * visual_cost + 0.3 * norm_distance
                cost_matrix[i, j] = cost
                
        # Hungarian algoritması
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Eşleşmeleri filtrele
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.8:  # Maliyet eşiği
                matches.append((trackers[r], detections[c]))
                
        return matches
        
    def _calculate_visual_similarity(self, tracker: DroneTrack, 
                                   feature: Optional[np.ndarray]) -> float:
        """Görsel benzerlik hesapla"""
        if feature is None:
            return 0.5  # Nötr
            
        if tracker.last_visual_feature is None:
            tracker.last_visual_feature = feature
            return 0.5  # İlk frame için nötr
            
        # Basit cosine similarity
        similarity = np.dot(tracker.last_visual_feature, feature)
        tracker.last_visual_feature = feature
        
        return float(similarity)
        
    def _init_track(self, detection: Dict):
        """Yeni drone takibi başlat"""
        track_id = self._generate_track_id()
        tracker = DroneTrack(track_id, detection)
        self.trackers.append(tracker)
        
    def _manage_tracks(self):
        """Takip yönetimi - mesafeye göre adaptif"""
        updated_trackers = []
        
        for tracker in self.trackers:
            # Mesafeye göre max_age belirle
            if self.range_adaptive:
                max_age = self.range_params[tracker.range_category]['max_age']
            else:
                max_age = self.max_age
                
            # Takip güvenilirliğine göre ek tolerans
            if tracker.tracking_confidence > 0.7:
                max_age = int(max_age * 1.5)
                
            if tracker.lost_frames < max_age:
                updated_trackers.append(tracker)
            else:
                logger.debug(f"Track {tracker.track_id} removed after {tracker.lost_frames} lost frames")
                
        self.trackers = updated_trackers
        
    def _state_to_bbox(self, state: np.ndarray) -> List[int]:
        """Kalman durumunu bbox'a çevir"""
        cx, cy, z, s, r = state[:5]
        
        # NaN veya sonsuz değer kontrolü
        if np.any(np.isnan([cx, cy, z, s, r])) or np.any(np.isinf([cx, cy, z, s, r])):
            # Son bilinen bbox'ı dön
            if hasattr(self, 'last_valid_bbox'):
                return self.last_valid_bbox
            else:
                return [100, 100, 200, 200]  # Varsayılan
        
        # Güvenli değerler
        s = max(s, 100.0)  # Minimum alan
        r = np.clip(r, 0.1, 10.0)  # Oran sınırları
        
        # Güvenli hesaplama
        w = np.sqrt(max(s * r, 100))
        h = s / w if w > 0 else np.sqrt(s)
        
        # Makul sınırlar
        w = np.clip(w, 10, 2000)
        h = np.clip(h, 10, 2000)
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        bbox = [x1, y1, x2, y2]
        self.last_valid_bbox = bbox
        
        return bbox