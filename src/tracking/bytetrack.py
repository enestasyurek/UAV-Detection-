# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# SciPy'den doğrusal atama için optimize modülünü içe aktar
from scipy.optimize import linear_sum_assignment
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# Temel takip sınıfını içe aktar
from .base_tracker import BaseTracker, Track
# Loglama için logging modülünü içe aktar
import logging

# Logger oluştur
logger = logging.getLogger(__name__)


class ByteTracker(BaseTracker):
    """
    ByteTrack algoritması implementasyonu
    Düşük güven skorlu tespitleri de değerlendiren gelişmiş takip algoritması
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 min_box_area: float = 100):
        """
        ByteTrack başlatıcı
        
        Args:
            track_thresh: Yüksek güven skoru eşiği
            track_buffer: Takip tamponu (frame sayısı)
            match_thresh: Eşleştirme eşiği
            min_box_area: Minimum kutu alanı
        """
        # Temel sınıfı başlat
        super().__init__()
        
        # ByteTrack parametreleri
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        # Takip listeleri
        self.tracked_tracks = []  # Aktif takipler
        self.lost_tracks = []     # Kayıp takipler
        self.removed_tracks = []  # Silinen takipler
        
        # Frame sayacı
        self.frame_id = 0
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        ByteTrack algoritması ile takip güncelleme
        
        Args:
            detections: YOLO tespitleri
            
        Returns:
            Takip ID'li tespitler
        """
        # Frame sayacını artır
        self.frame_id += 1
        
        # Boş tespit kontrolü
        if not detections:
            # Tüm takipleri kayıp olarak işaretle
            for track in self.tracked_tracks:
                track.mark_lost()
            # Kayıp takipleri yönet
            self._manage_lost_tracks()
            return []
        
        # Tespitleri yüksek ve düşük güven skoruna göre ayır
        high_det, low_det = self._split_detections(detections)
        
        # Mevcut takipleri al
        tracked_bbox = [track.get_latest_bbox() for track in self.tracked_tracks]
        
        # İlk aşama: Yüksek skorlu tespitlerle aktif takipleri eşleştir
        matches1, unmatched_tracks1, unmatched_dets1 = self._match_detections(
            tracked_bbox, high_det, thresh=self.match_thresh
        )
        
        # Eşleşen takipleri güncelle
        for track_idx, det_idx in matches1:
            self.tracked_tracks[track_idx].update(high_det[det_idx])
        
        # İkinci aşama: Eşleşmeyen takipleri düşük skorlu tespitlerle eşleştir
        if unmatched_tracks1 and low_det:
            remaining_tracks = [self.tracked_tracks[i] for i in unmatched_tracks1]
            remaining_bbox = [track.get_latest_bbox() for track in remaining_tracks]
            
            matches2, unmatched_tracks2, unmatched_dets2 = self._match_detections(
                remaining_bbox, low_det, thresh=0.5
            )
            
            # İkinci aşama eşleşmelerini güncelle
            for track_idx, det_idx in matches2:
                actual_track_idx = unmatched_tracks1[track_idx]
                self.tracked_tracks[actual_track_idx].update(low_det[det_idx])
                
            # Hala eşleşmeyen takipleri kayıp olarak işaretle
            for idx in unmatched_tracks2:
                actual_idx = unmatched_tracks1[idx]
                self.tracked_tracks[actual_idx].mark_lost()
        else:
            # Eşleşmeyen takipleri kayıp olarak işaretle
            for idx in unmatched_tracks1:
                self.tracked_tracks[idx].mark_lost()
        
        # Üçüncü aşama: Kayıp takipleri kalan tespitlerle eşleştir
        if self.lost_tracks and unmatched_dets1:
            lost_bbox = [track.get_latest_bbox() for track in self.lost_tracks]
            remaining_high_det = [high_det[i] for i in unmatched_dets1]
            
            matches3, unmatched_lost, unmatched_dets3 = self._match_detections(
                lost_bbox, remaining_high_det, thresh=self.match_thresh
            )
            
            # Kayıp takipleri yeniden aktifleştir
            reactivated = []
            for track_idx, det_idx in matches3:
                track = self.lost_tracks[track_idx]
                actual_det_idx = unmatched_dets1[det_idx]
                track.update(high_det[actual_det_idx])
                track.lost_frames = 0
                reactivated.append(track)
                
            # Yeniden aktifleştirilen takipleri kayıp listesinden çıkar
            for track in reactivated:
                self.lost_tracks.remove(track)
                self.tracked_tracks.append(track)
                
            # Kalan eşleşmeyen tespitler için yeni takip başlat
            for idx in unmatched_dets3:
                actual_idx = unmatched_dets1[idx]
                self._init_new_track(high_det[actual_idx])
        else:
            # Eşleşmeyen yüksek skorlu tespitler için yeni takip başlat
            for idx in unmatched_dets1:
                self._init_new_track(high_det[idx])
        
        # Kayıp takipleri yönet
        self._manage_lost_tracks()
        
        # Sonuçları hazırla
        results = []
        for track in self.tracked_tracks:
            if track.lost_frames == 0:  # Sadece aktif takipler
                detection = {
                    'bbox': track.get_latest_bbox(),
                    'center': track.get_latest_center(),
                    'confidence': track.confidence_history[-1],
                    'class': track.class_name,
                    'track_id': track.track_id,
                    'age': track.age
                }
                results.append(detection)
                
        return results
    
    def _split_detections(self, detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Tespitleri güven skoruna göre ayır"""
        high_det = []
        low_det = []
        
        for det in detections:
            # Kutu alanını hesapla
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Minimum alan kontrolü
            if area < self.min_box_area:
                continue
                
            # Güven skoruna göre ayır
            if det['confidence'] >= self.track_thresh:
                high_det.append(det)
            else:
                low_det.append(det)
                
        return high_det, low_det
    
    def _match_detections(self, 
                         tracked_bbox: List[List[int]], 
                         detections: List[Dict],
                         thresh: float) -> Tuple[List[Tuple], List[int], List[int]]:
        """IoU tabanlı tespit eşleştirme"""
        
        # Boş liste kontrolü
        if not tracked_bbox or not detections:
            return [], list(range(len(tracked_bbox))), list(range(len(detections)))
        
        # IoU matrisini hesapla
        iou_matrix = np.zeros((len(tracked_bbox), len(detections)))
        
        for i, t_bbox in enumerate(tracked_bbox):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(t_bbox, det['bbox'])
        
        # Maliyet matrisini oluştur (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Hungarian algoritması ile eşleştirme
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Eşleşmeleri filtrele (eşik kontrolü)
        matches = []
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= thresh:
                matches.append((i, j))
        
        # Eşleşmeyen takip ve tespitleri bul
        unmatched_tracks = []
        unmatched_dets = []
        
        for i in range(len(tracked_bbox)):
            if i not in [m[0] for m in matches]:
                unmatched_tracks.append(i)
                
        for j in range(len(detections)):
            if j not in [m[1] for m in matches]:
                unmatched_dets.append(j)
                
        return matches, unmatched_tracks, unmatched_dets
    
    def _init_new_track(self, detection: Dict):
        """Yeni takip başlat"""
        # Yeni takip ID'si oluştur
        track_id = self._generate_track_id()
        # Yeni takip nesnesi oluştur
        new_track = Track(track_id, detection)
        # Aktif takip listesine ekle
        self.tracked_tracks.append(new_track)
        # Takip sözlüğüne ekle
        self.tracks[track_id] = {
            'track': new_track,
            'lost_frames': 0
        }
        
    def _manage_lost_tracks(self):
        """Kayıp takipleri yönet"""
        # Silinecek takipleri belirle
        to_remove = []
        
        for track in self.tracked_tracks:
            if track.lost_frames > 0:
                # Kayıp takip listesine ekle
                self.lost_tracks.append(track)
                to_remove.append(track)
                
        # Aktif listeden çıkar
        for track in to_remove:
            self.tracked_tracks.remove(track)
            
        # Çok uzun süredir kayıp olanları sil
        self.lost_tracks = [t for t in self.lost_tracks 
                           if t.lost_frames <= self.track_buffer]
        
    def get_tracking_stats(self) -> Dict:
        """Takip istatistiklerini döndür"""
        return {
            'active_tracks': len(self.tracked_tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_tracks': len(self.tracks),
            'frame_id': self.frame_id
        }