# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# SciPy'den doğrusal atama için optimize modülünü içe aktar
from scipy.optimize import linear_sum_assignment
# Kalman filtresi için filterpy kütüphanesinden içe aktar
from filterpy.kalman import KalmanFilter
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# Temel takip sınıfını içe aktar
from .base_tracker import BaseTracker, Track
# Loglama için logging modülünü içe aktar
import logging

# Logger oluştur
logger = logging.getLogger(__name__)


class OCSortTrack(Track):
    """OC-SORT için genişletilmiş takip sınıfı"""
    
    def __init__(self, track_id: str, initial_detection: Dict):
        """
        OC-SORT takip nesnesi
        
        Args:
            track_id: Takip ID'si
            initial_detection: İlk tespit
        """
        # Temel sınıfı başlat
        super().__init__(track_id, initial_detection)
        
        # Kalman filtresi (7 durum: x, y, s, r, dx, dy, ds)
        # s: alan (scale), r: en-boy oranı (aspect ratio)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self._init_kalman_filter(initial_detection['bbox'])
        
        # Gözlem geçmişi (OC-SORT'un ana yeniliği)
        self.observation_history = []
        self.history_size = 30
        
        # Hız tahmini için
        self.velocity_history = []
        self.smooth_velocity = np.zeros(2)
        
        # Geçerli bbox sakla
        self.last_valid_bbox = initial_detection['bbox']
        
    def _init_kalman_filter(self, bbox: List[int]):
        """Kalman filtresini başlat"""
        # Kutuyu merkez, alan ve en-boy oranına çevir
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)  # alan
        r = (x2 - x1) / (y2 - y1)  # en-boy oranı
        
        # Başlangıç durumu [x, y, s, r, dx, dy, ds]
        self.kf.x = np.array([cx, cy, s, r, 0, 0, 0])
        
        # Durum geçiş matrisi
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Ölçüm matrisi
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Belirsizlik matrisleri
        self.kf.R[2:, 2:] *= 10.0  # Alan ve oran için yüksek belirsizlik
        self.kf.P[4:, 4:] *= 1000.0  # Hız için yüksek başlangıç belirsizliği
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
    def predict(self) -> np.ndarray:
        """Kalman tahmini yap"""
        # Hız düzeltmesi (OC-SORT özelliği)
        if self.age > 1:
            # Smooth velocity kullan
            self.kf.x[4] = self.smooth_velocity[0]
            self.kf.x[5] = self.smooth_velocity[1]
            
        self.kf.predict()
        return self._state_to_bbox(self.kf.x)
        
    def update_kalman(self, bbox: List[int]):
        """Kalman filtresini güncelle"""
        # Ölçümü hazırla
        measurement = self._bbox_to_measurement(bbox)
        
        # Gözlem geçmişine ekle
        self.observation_history.append(measurement)
        if len(self.observation_history) > self.history_size:
            self.observation_history.pop(0)
            
        # Kalman güncellemesi
        self.kf.update(measurement)
        
        # Hız hesapla ve smooth et
        if len(self.observation_history) >= 2:
            # Son iki gözlemden hız hesapla
            dx = self.observation_history[-1][0] - self.observation_history[-2][0]
            dy = self.observation_history[-1][1] - self.observation_history[-2][1]
            
            self.velocity_history.append([dx, dy])
            if len(self.velocity_history) > 5:
                self.velocity_history.pop(0)
                
            # Hızı smooth et (hareketli ortalama)
            if self.velocity_history:
                self.smooth_velocity = np.mean(self.velocity_history, axis=0)
                
    def _bbox_to_measurement(self, bbox: List[int]) -> np.ndarray:
        """Sınırlayıcı kutuyu Kalman ölçümüne çevir"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = max((x2 - x1) * (y2 - y1), 1)  # Minimum alan: 1
        r = (x2 - x1) / max(y2 - y1, 1)  # Bölme hatasını önle
        return np.array([cx, cy, s, r])
        
    def _state_to_bbox(self, state: np.ndarray) -> List[int]:
        """Kalman durumunu sınırlayıcı kutuya çevir"""
        cx, cy, s, r = state[:4]
        
        # NaN veya sonsuz değer kontrolü
        if np.isnan(cx) or np.isnan(cy) or np.isnan(s) or np.isnan(r) or \
           np.isinf(cx) or np.isinf(cy) or np.isinf(s) or np.isinf(r):
            # Son bilinen bbox'ı dön
            if hasattr(self, 'last_valid_bbox'):
                return self.last_valid_bbox
            else:
                return [100, 100, 200, 200]  # Varsayılan
        
        # Negatif veya sıfır değer kontrolü
        s = max(s, 100.0)  # Minimum alan: 100 piksel kare
        r = max(r, 0.1)    # Minimum oran: 0.1
        r = min(r, 10.0)   # Maximum oran: 10.0
        
        # Güvenli sqrt hesaplama
        try:
            w = np.sqrt(s * r)
            h = s / w if w > 0 else np.sqrt(s)
        except:
            # Hata durumunda varsayılan boyutlar
            w = h = np.sqrt(max(s, 100))
        
        # Makul sınırlar içinde tut
        w = np.clip(w, 10, 2000)  # 10-2000 piksel arası
        h = np.clip(h, 10, 2000)  # 10-2000 piksel arası
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        bbox = [x1, y1, x2, y2]
        self.last_valid_bbox = bbox  # Geçerli bbox'ı sakla
        
        return bbox
        
    def get_observation_confidence(self) -> float:
        """Gözlem güvenilirliğini hesapla"""
        # Gözlem sayısına dayalı güven
        if not self.observation_history:
            return 0.0
            
        # Son gözlemler arası tutarlılık
        if len(self.observation_history) < 2:
            return 0.5
            
        # Son 5 gözlemin varyansı
        recent_obs = self.observation_history[-5:]
        if len(recent_obs) < 2:
            return 0.5
            
        # Pozisyon varyansı
        positions = np.array([[obs[0], obs[1]] for obs in recent_obs])
        variance = np.var(positions, axis=0).mean()
        
        # Düşük varyans = yüksek güven
        confidence = 1.0 / (1.0 + variance / 100.0)
        return confidence


class OCSortTracker(BaseTracker):
    """
    OC-SORT (Observation-Centric SORT) algoritması implementasyonu
    Gözlem geçmişi ve hız düzeltmesi kullanan gelişmiş takip
    """
    
    def __init__(self,
                 det_thresh: float = 0.2,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 use_byte: bool = True):
        """
        OC-SORT başlatıcı
        
        Args:
            det_thresh: Tespit eşiği
            max_age: Maksimum kayıp frame
            min_hits: Minimum onay için görülme
            iou_threshold: IoU eşiği
            delta_t: Hız hesaplama frame aralığı
            use_byte: ByteTrack benzeri düşük skor kullanımı
        """
        # Temel sınıfı başlat
        super().__init__()
        
        # OC-SORT parametreleri
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.use_byte = use_byte
        
        # Takip listeleri
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        OC-SORT ile takip güncelleme
        
        Args:
            detections: YOLO tespitleri
            
        Returns:
            Takip ID'li tespitler
        """
        # Frame sayacını artır
        self.frame_count += 1
        
        # Tahminleri yap
        for tracker in self.trackers:
            tracker.predict()
            
        # Tespitleri yüksek ve düşük skora ayır
        if self.use_byte:
            high_det = [d for d in detections if d['confidence'] >= self.det_thresh]
            low_det = [d for d in detections if d['confidence'] < self.det_thresh]
        else:
            high_det = detections
            low_det = []
            
        # İlk aşama: Yüksek skorlu tespitlerle eşleştir
        matched1, unmatched_trackers1, unmatched_dets1 = self._associate(
            self.trackers, high_det, self.iou_threshold
        )
        
        # Eşleşen takipleri güncelle
        for t_idx, d_idx in matched1:
            self.trackers[t_idx].update(high_det[d_idx])
            self.trackers[t_idx].update_kalman(high_det[d_idx]['bbox'])
            
        # İkinci aşama: Eşleşmeyen takipleri düşük skorlu tespitlerle eşleştir
        if self.use_byte and unmatched_trackers1 and low_det:
            remaining_trackers = [self.trackers[i] for i in unmatched_trackers1]
            
            # Gözlem güvenilirliğine göre dinamik IoU eşiği
            dynamic_ious = []
            for tracker in remaining_trackers:
                confidence = tracker.get_observation_confidence()
                # Yüksek güvenilirlik = daha esnek eşleştirme
                dynamic_iou = self.iou_threshold * (2.0 - confidence)
                dynamic_ious.append(min(dynamic_iou, 0.9))
                
            matched2, unmatched_trackers2, unmatched_dets2 = self._associate_dynamic(
                remaining_trackers, low_det, dynamic_ious
            )
            
            # İkinci aşama güncellemeleri
            for t_idx, d_idx in matched2:
                actual_t_idx = unmatched_trackers1[t_idx]
                self.trackers[actual_t_idx].update(low_det[d_idx])
                self.trackers[actual_t_idx].update_kalman(low_det[d_idx]['bbox'])
                
            # Güncel eşleşmeyenler
            final_unmatched_trackers = [unmatched_trackers1[i] for i in unmatched_trackers2]
        else:
            final_unmatched_trackers = unmatched_trackers1
            
        # Eşleşmeyen takipleri kayıp olarak işaretle
        for idx in final_unmatched_trackers:
            self.trackers[idx].mark_lost()
            
        # Üçüncü aşama: Yeni takipler oluştur
        for idx in unmatched_dets1:
            self._init_track(high_det[idx])
            
        # Dördüncü aşama: Gözlem geçmişi yeniden ilişkilendirme (OC-SORT özelliği)
        lost_trackers = [t for t in self.trackers if t.lost_frames > 0]
        if lost_trackers and unmatched_dets1:
            # Geçmiş gözlemlere dayalı yeniden ilişkilendirme
            remaining_dets = [high_det[i] for i in unmatched_dets1]
            matched3 = self._associate_with_history(lost_trackers, remaining_dets)
            
            for t, d in matched3:
                t.update(d)
                t.update_kalman(d['bbox'])
                t.lost_frames = 0
                
        # Eski takipleri temizle
        self.trackers = [t for t in self.trackers if t.lost_frames < self.max_age]
        
        # Sonuçları hazırla
        results = []
        for tracker in self.trackers:
            # Yeterli onay ve aktif takipler
            if tracker.age >= self.min_hits and tracker.lost_frames == 0:
                bbox = tracker.get_latest_bbox()
                detection = {
                    'bbox': bbox,
                    'center': tracker.get_latest_center(),
                    'confidence': tracker.confidence_history[-1],
                    'class': tracker.class_name,
                    'track_id': tracker.track_id,
                    'age': tracker.age,
                    'observation_confidence': tracker.get_observation_confidence()
                }
                results.append(detection)
                
        return results
        
    def _associate(self, trackers: List[OCSortTrack], 
                   detections: List[Dict],
                   iou_thresh: float) -> Tuple:
        """Standart IoU tabanlı ilişkilendirme"""
        
        if not trackers or not detections:
            return [], list(range(len(trackers))), list(range(len(detections)))
            
        # IoU matrisi hesapla
        iou_matrix = np.zeros((len(trackers), len(detections)))
        
        for t_idx, tracker in enumerate(trackers):
            # Tahmin edilen kutuyu kullan
            pred_bbox = self._state_to_bbox(tracker.kf.x)
            
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._calculate_iou(pred_bbox, det['bbox'])
                
        # Maliyet matrisi (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Hungarian algoritması
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Eşik kontrolü ile filtrele
        matched = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresh:
                matched.append((r, c))
                
        # Eşleşmeyenleri bul
        unmatched_trackers = list(set(range(len(trackers))) - set([m[0] for m in matched]))
        unmatched_dets = list(set(range(len(detections))) - set([m[1] for m in matched]))
        
        return matched, unmatched_trackers, unmatched_dets
        
    def _associate_dynamic(self, trackers: List[OCSortTrack],
                          detections: List[Dict],
                          iou_thresholds: List[float]) -> Tuple:
        """Dinamik IoU eşikleri ile ilişkilendirme"""
        
        if not trackers or not detections:
            return [], list(range(len(trackers))), list(range(len(detections)))
            
        # IoU matrisi
        iou_matrix = np.zeros((len(trackers), len(detections)))
        
        for t_idx, tracker in enumerate(trackers):
            pred_bbox = self._state_to_bbox(tracker.kf.x)
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._calculate_iou(pred_bbox, det['bbox'])
                
        # Maliyet matrisi
        cost_matrix = 1 - iou_matrix
        
        # Hungarian algoritması
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Dinamik eşik kontrolü
        matched = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresholds[r]:
                matched.append((r, c))
                
        # Eşleşmeyenler
        unmatched_trackers = list(set(range(len(trackers))) - set([m[0] for m in matched]))
        unmatched_dets = list(set(range(len(detections))) - set([m[1] for m in matched]))
        
        return matched, unmatched_trackers, unmatched_dets
        
    def _associate_with_history(self, lost_trackers: List[OCSortTrack],
                               detections: List[Dict]) -> List[Tuple]:
        """Gözlem geçmişi kullanarak yeniden ilişkilendirme"""
        matched = []
        
        for tracker in lost_trackers:
            if len(tracker.observation_history) < 3:
                continue
                
            # Son gözlemlere dayalı tahmin
            best_iou = 0
            best_det = None
            
            for det in detections:
                # Geçmiş gözlemlere göre ekstrapolasyon
                if tracker.lost_frames <= 3:
                    # Yakın geçmiş için basit tahmin
                    pred_bbox = self._state_to_bbox(tracker.kf.x)
                else:
                    # Uzak geçmiş için hız tabanlı tahmin
                    pred_center = tracker.predict_next_position()
                    last_bbox = tracker.get_latest_bbox()
                    w = last_bbox[2] - last_bbox[0]
                    h = last_bbox[3] - last_bbox[1]
                    pred_bbox = [
                        int(pred_center[0] - w/2),
                        int(pred_center[1] - h/2),
                        int(pred_center[0] + w/2),
                        int(pred_center[1] + h/2)
                    ]
                    
                iou = self._calculate_iou(pred_bbox, det['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_det = det
                    
            if best_det:
                matched.append((tracker, best_det))
                detections.remove(best_det)
                
        return matched
        
    def _init_track(self, detection: Dict):
        """Yeni takip başlat"""
        track_id = self._generate_track_id()
        tracker = OCSortTrack(track_id, detection)
        self.trackers.append(tracker)
        
    def _state_to_bbox(self, state: np.ndarray) -> List[int]:
        """Kalman durumunu sınırlayıcı kutuya çevir"""
        cx, cy, s, r = state[:4]
        
        # NaN veya sonsuz değer kontrolü
        if np.isnan(cx) or np.isnan(cy) or np.isnan(s) or np.isnan(r) or \
           np.isinf(cx) or np.isinf(cy) or np.isinf(s) or np.isinf(r):
            # Son bilinen bbox'ı dön
            if hasattr(self, 'last_valid_bbox'):
                return self.last_valid_bbox
            else:
                return [100, 100, 200, 200]  # Varsayılan
        
        # Negatif veya sıfır değer kontrolü
        s = max(s, 100.0)  # Minimum alan: 100 piksel kare
        r = max(r, 0.1)    # Minimum oran: 0.1
        r = min(r, 10.0)   # Maximum oran: 10.0
        
        # Güvenli sqrt hesaplama
        try:
            w = np.sqrt(s * r)
            h = s / w if w > 0 else np.sqrt(s)
        except:
            # Hata durumunda varsayılan boyutlar
            w = h = np.sqrt(max(s, 100))
        
        # Makul sınırlar içinde tut
        w = np.clip(w, 10, 2000)  # 10-2000 piksel arası
        h = np.clip(h, 10, 2000)  # 10-2000 piksel arası
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        bbox = [x1, y1, x2, y2]
        self.last_valid_bbox = bbox  # Geçerli bbox'ı sakla
        
        return bbox