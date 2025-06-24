# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# PyTorch kütüphanesini derin öğrenme için içe aktar
import torch
import torch.nn as nn
# Torchvision'dan özellik çıkarıcı modeller için içe aktar
import torchvision.transforms as transforms
import torchvision.models as models
# SciPy'den mesafe hesaplamaları için içe aktar
from scipy.spatial.distance import cosine
# Kalman filtresi için filterpy kütüphanesinden içe aktar
from filterpy.kalman import KalmanFilter
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# Temel takip sınıfını içe aktar
from .base_tracker import BaseTracker, Track
# OpenCV'yi görüntü işleme için içe aktar
import cv2
# Loglama için logging modülünü içe aktar
import logging

# Logger oluştur
logger = logging.getLogger(__name__)


class DeepSortTrack(Track):
    """DeepSORT için genişletilmiş takip sınıfı"""
    
    def __init__(self, track_id: str, initial_detection: Dict, initial_feature: np.ndarray):
        """
        DeepSORT takip nesnesi
        
        Args:
            track_id: Takip ID'si
            initial_detection: İlk tespit
            initial_feature: İlk görünüm özelliği
        """
        # Temel sınıfı başlat
        super().__init__(track_id, initial_detection)
        
        # Görünüm özellikleri
        self.features = [initial_feature]
        self.feature_history_size = 100
        
        # Kalman filtresi başlat (4 durum: x, y, vx, vy)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self._init_kalman_filter(initial_detection['center'])
        
    def _init_kalman_filter(self, initial_center: Tuple[int, int]):
        """Kalman filtresini başlat"""
        # Başlangıç durumu [x, y, vx, vy]
        self.kf.x = np.array([initial_center[0], initial_center[1], 0., 0.])
        
        # Durum geçiş matrisi
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        # Ölçüm matrisi
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        # Ölçüm belirsizliği
        self.kf.R *= 10.0
        
        # Süreç belirsizliği
        self.kf.Q[2:, 2:] *= 0.01
        self.kf.Q[0:2, 0:2] *= 0.1
        
        # Başlangıç kovaryansı
        self.kf.P *= 10.0
        
    def predict(self):
        """Kalman filtresi ile sonraki pozisyonu tahmin et"""
        self.kf.predict()
        return self.kf.x[:2]  # [x, y] döndür
        
    def update_kalman(self, center: Tuple[int, int]):
        """Kalman filtresini güncelle"""
        measurement = np.array([center[0], center[1]])
        self.kf.update(measurement)
        
    def add_feature(self, feature: np.ndarray):
        """Yeni görünüm özelliği ekle"""
        self.features.append(feature)
        # Özellik geçmişi boyutunu sınırla
        if len(self.features) > self.feature_history_size:
            self.features.pop(0)
            
    def get_feature_vector(self) -> np.ndarray:
        """Ortalama özellik vektörünü döndür"""
        if not self.features:
            return np.zeros(128)  # Varsayılan boyut
        return np.mean(self.features, axis=0)


class ReIDNet(nn.Module):
    """Yeniden tanımlama (Re-ID) için özellik çıkarıcı ağ"""
    
    def __init__(self, feature_dim: int = 128):
        """
        Re-ID ağı başlatıcı
        
        Args:
            feature_dim: Özellik vektörü boyutu
        """
        super(ReIDNet, self).__init__()
        
        # ResNet18'i temel olarak kullan
        resnet = models.resnet18(pretrained=True)
        
        # Son katmanı çıkar
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Özellik projeksiyon katmanı
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Görüntü ön işleme
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standart person Re-ID boyutu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        """İleri yayılım"""
        # Backbone'den özellik çıkar
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # Projeksiyon uygula
        features = self.projection(features)
        # L2 normalizasyon
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features
        
    def extract_features(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Görüntüden özellik çıkar"""
        # Sınırlayıcı kutuyu kırp
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]
        
        # Boş görüntü kontrolü
        if cropped.size == 0:
            return np.zeros(128)
            
        # Ön işleme uygula
        input_tensor = self.transform(cropped).unsqueeze(0)
        
        # Özellik çıkar
        with torch.no_grad():
            features = self.forward(input_tensor)
            
        return features.cpu().numpy().squeeze()


class DeepSortTracker(BaseTracker):
    """
    DeepSORT algoritması implementasyonu
    Kalman filtresi ve derin özellik eşleştirmesi kullanan gelişmiş takip
    """
    
    def __init__(self,
                 max_dist: float = 0.2,
                 min_confidence: float = 0.3,
                 max_iou_distance: float = 0.7,
                 max_age: int = 30,
                 n_init: int = 3):
        """
        DeepSORT başlatıcı
        
        Args:
            max_dist: Maksimum kosinus mesafesi
            min_confidence: Minimum güven skoru
            max_iou_distance: Maksimum IoU mesafesi
            max_age: Maksimum takip yaşı
            n_init: Onaylanmış takip için minimum görülme
        """
        # Temel sınıfı başlat
        super().__init__()
        
        # DeepSORT parametreleri
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        
        # Re-ID ağını başlat - LIGHTWEIGHT MOD
        self.use_reid = False  # Performans için Re-ID kapalı
        if self.use_reid:
            self.reid_net = ReIDNet()
            self.reid_net.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.reid_net.to(self.device)
        else:
            self.device = 'cpu'
        
        # Takip listeleri
        self.confirmed_tracks = []  # Onaylanmış takipler
        self.unconfirmed_tracks = []  # Onaylanmamış takipler
        
        logger.info(f"DeepSORT initialized on {self.device}")
        
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        DeepSORT ile takip güncelleme
        
        Args:
            detections: YOLO tespitleri
            frame: Mevcut görüntü frame'i
            
        Returns:
            Takip ID'li tespitler
        """
        # Güven skoru filtresi
        detections = [d for d in detections if d['confidence'] >= self.min_confidence]
        
        # Tüm takipler için Kalman tahmini yap
        for track in self.confirmed_tracks + self.unconfirmed_tracks:
            track.predict()
            
        # Tespitler için özellik çıkar
        detection_features = []
        if self.use_reid:
            for det in detections:
                feature = self.reid_net.extract_features(frame, det['bbox'])
                detection_features.append(feature)
        else:
            # Dummy features for speed
            for det in detections:
                detection_features.append(np.random.randn(128))
            
        # Onaylanmış takipleri eşleştir
        matched1, unmatched_tracks1, unmatched_dets1 = self._match_cascade(
            self.confirmed_tracks, detections, detection_features
        )
        
        # Eşleşmeyen onaylanmış takipleri IoU ile tekrar eşleştir
        remaining_tracks = [self.confirmed_tracks[i] for i in unmatched_tracks1]
        remaining_dets = [detections[i] for i in unmatched_dets1]
        remaining_features = [detection_features[i] for i in unmatched_dets1]
        
        matched2, unmatched_tracks2, unmatched_dets2 = self._match_iou(
            remaining_tracks, remaining_dets
        )
        
        # Tüm eşleşmeleri birleştir
        matches = matched1
        for track_idx, det_idx in matched2:
            actual_track_idx = unmatched_tracks1[track_idx]
            actual_det_idx = unmatched_dets1[det_idx]
            matches.append((actual_track_idx, actual_det_idx))
            
        # Eşleşen takipleri güncelle
        for track_idx, det_idx in matches:
            track = self.confirmed_tracks[track_idx]
            detection = detections[det_idx]
            feature = detection_features[det_idx]
            
            track.update(detection)
            track.update_kalman(detection['center'])
            if self.use_reid:
                track.add_feature(feature)
            else:
                track.add_feature(np.random.randn(128))
            
        # Onaylanmamış takipleri eşleştir
        unmatched_dets_final = []
        for i in unmatched_dets1:
            if i not in [m[1] for m in matched2]:
                unmatched_dets_final.append(i)
                
        matched3, unmatched_unconf, unmatched_dets3 = self._match_cascade(
            self.unconfirmed_tracks, 
            [detections[i] for i in unmatched_dets_final],
            [detection_features[i] for i in unmatched_dets_final]
        )
        
        # Onaylanmamış takipleri güncelle
        for track_idx, det_idx in matched3:
            track = self.unconfirmed_tracks[track_idx]
            actual_det_idx = unmatched_dets_final[det_idx]
            
            track.update(detections[actual_det_idx])
            track.update_kalman(detections[actual_det_idx]['center'])
            if self.use_reid:
                track.add_feature(detection_features[actual_det_idx])
            else:
                track.add_feature(np.random.randn(128))
            
            # Yeterli görülme sayısına ulaştıysa onaylı listeye taşı
            if track.age >= self.n_init:
                self.confirmed_tracks.append(track)
                self.unconfirmed_tracks.remove(track)
                
        # Yeni takipler oluştur
        for det_idx in unmatched_dets3:
            actual_idx = unmatched_dets_final[det_idx]
            self._init_track(detections[actual_idx], detection_features[actual_idx])
            
        # Kayıp takipleri işle
        self._mark_lost_tracks(unmatched_tracks1, unmatched_tracks2, unmatched_unconf)
        
        # Eski takipleri temizle
        self._clean_tracks()
        
        # Sonuçları hazırla
        results = []
        for track in self.confirmed_tracks:
            if track.lost_frames == 0:
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
        
    def _match_cascade(self, tracks: List[DeepSortTrack], 
                      detections: List[Dict],
                      features: List[np.ndarray]) -> Tuple:
        """Kaskad eşleştirme (görünüm + hareket)"""
        
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
            
        # Maliyet matrisi hesapla
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_feature = track.get_feature_vector()
            
            for j, det_feature in enumerate(features):
                # Görünüm mesafesi (kosinus)
                if self.use_reid:
                    appearance_dist = cosine(track_feature, det_feature)
                else:
                    # Lightweight mode - sadece mesafe bazlı
                    appearance_dist = 0.5  # Sabit orta değer
                
                # Hareket mesafesi (Mahalanobis)
                motion_dist = self._gating_distance(track, detections[j])
                
                # Kombine maliyet
                if motion_dist > self.max_iou_distance:
                    cost_matrix[i, j] = 1e5  # Çok yüksek maliyet
                else:
                    cost_matrix[i, j] = appearance_dist
                    
        # Hungarian algoritması ile eşleştir
        matches, unmatched_tracks, unmatched_dets = self._hungarian_match(
            cost_matrix, self.max_dist
        )
        
        return matches, unmatched_tracks, unmatched_dets
        
    def _match_iou(self, tracks: List[DeepSortTrack], 
                   detections: List[Dict]) -> Tuple:
        """IoU tabanlı eşleştirme"""
        
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
            
        # IoU maliyet matrisi
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._calculate_iou(track.get_latest_bbox(), det['bbox'])
                cost_matrix[i, j] = 1 - iou
                
        # Eşleştir
        matches, unmatched_tracks, unmatched_dets = self._hungarian_match(
            cost_matrix, 1 - self.max_iou_distance
        )
        
        return matches, unmatched_tracks, unmatched_dets
        
    def _hungarian_match(self, cost_matrix: np.ndarray, 
                        thresh: float) -> Tuple:
        """Hungarian algoritması ile eşleştirme"""
        from scipy.optimize import linear_sum_assignment
        
        # Optimum atama
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Eşik kontrolü ile filtreleme
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= thresh:
                matches.append((row, col))
                
        # Eşleşmeyenleri bul
        unmatched_rows = list(set(range(cost_matrix.shape[0])) - set([m[0] for m in matches]))
        unmatched_cols = list(set(range(cost_matrix.shape[1])) - set([m[1] for m in matches]))
        
        return matches, unmatched_rows, unmatched_cols
        
    def _gating_distance(self, track: DeepSortTrack, detection: Dict) -> float:
        """Mahalanobis mesafesi hesapla"""
        # Basitleştirilmiş versiyon - sadece Öklid mesafesi
        predicted = track.kf.x[:2]
        measured = np.array(detection['center'])
        distance = np.linalg.norm(predicted - measured)
        
        # Normalize et
        return distance / 100.0
        
    def _init_track(self, detection: Dict, feature: np.ndarray):
        """Yeni takip başlat"""
        track_id = self._generate_track_id()
        if self.use_reid:
            track = DeepSortTrack(track_id, detection, feature)
        else:
            # Dummy feature for lightweight mode
            track = DeepSortTrack(track_id, detection, np.random.randn(128))
        self.unconfirmed_tracks.append(track)
        
    def _mark_lost_tracks(self, *unmatched_indices):
        """Eşleşmeyen takipleri kayıp olarak işaretle"""
        # Tüm eşleşmeyen indeksleri topla
        all_unmatched = []
        for indices in unmatched_indices:
            all_unmatched.extend(indices)
            
        # Kayıp olarak işaretle
        for track in self.confirmed_tracks + self.unconfirmed_tracks:
            if track not in all_unmatched:
                track.mark_lost()
                
    def _clean_tracks(self):
        """Eski takipleri temizle"""
        # Onaylanmış takipleri temizle
        self.confirmed_tracks = [t for t in self.confirmed_tracks 
                                if t.lost_frames <= self.max_age]
        
        # Onaylanmamış takipleri daha agresif temizle
        self.unconfirmed_tracks = [t for t in self.unconfirmed_tracks 
                                  if t.lost_frames <= 3]