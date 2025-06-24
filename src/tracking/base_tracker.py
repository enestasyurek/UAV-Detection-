# ABC (Abstract Base Classes) modülünü soyut sınıflar için içe aktar
from abc import ABC, abstractmethod
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Dict, Tuple, Optional
# UUID oluşturmak için uuid modülünü içe aktar
import uuid


class BaseTracker(ABC):
    """Tüm takip algoritmaları için temel soyut sınıf"""
    
    def __init__(self):
        """Temel takip sınıfı başlatıcı"""
        # Takip edilen nesneleri saklamak için sözlük
        self.tracks = {}
        # Benzersiz takip ID'si sayacı
        self.track_id_counter = 0
        # Kayıp takip eşiği (kaç frame sonra takip silinir)
        self.max_lost_frames = 30
        
    @abstractmethod
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Tespitleri güncelle ve takip et
        
        Args:
            detections: YOLO'dan gelen tespit listesi
            
        Returns:
            Takip ID'leri eklenmiş tespit listesi
        """
        pass
    
    def _generate_track_id(self) -> str:
        """Benzersiz takip ID'si oluştur"""
        # Sayacı artır
        self.track_id_counter += 1
        # Benzersiz ID döndür
        return f"track_{self.track_id_counter}"
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        İki kutu arasındaki IoU (Intersection over Union) hesapla
        
        Args:
            box1: İlk kutu [x1, y1, x2, y2]
            box2: İkinci kutu [x1, y1, x2, y2]
            
        Returns:
            IoU değeri (0-1 arası)
        """
        # Kutu koordinatlarını al
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Kesişim alanının koordinatlarını hesapla
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Kesişim alanını hesapla
        if x2_i < x1_i or y2_i < y1_i:
            # Kesişim yoksa 0 döndür
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Her iki kutunun alanını hesapla
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Birleşim alanını hesapla
        union = area1 + area2 - intersection
        
        # IoU'yu hesapla ve döndür
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """
        İki merkez noktası arasındaki Öklid mesafesini hesapla
        
        Args:
            center1: İlk merkez (x, y)
            center2: İkinci merkez (x, y)
            
        Returns:
            Öklid mesafesi
        """
        # X ve Y farklarını hesapla
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        # Öklid mesafesini hesapla ve döndür
        return np.sqrt(dx**2 + dy**2)
    
    def clean_lost_tracks(self):
        """Kayıp takipleri temizle"""
        # Silinecek takipleri belirle
        tracks_to_delete = []
        
        # Her takibi kontrol et
        for track_id, track_info in self.tracks.items():
            # Eğer kayıp frame sayısı eşiği aştıysa
            if track_info.get('lost_frames', 0) > self.max_lost_frames:
                # Silinecekler listesine ekle
                tracks_to_delete.append(track_id)
        
        # Belirlenen takipleri sil
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            
    def get_active_tracks(self) -> Dict:
        """Aktif takipleri döndür"""
        # Sadece kayıp olmayan takipleri filtrele
        active_tracks = {}
        for track_id, track_info in self.tracks.items():
            if track_info.get('lost_frames', 0) == 0:
                active_tracks[track_id] = track_info
        return active_tracks


class Track:
    """Tek bir takip edilen nesneyi temsil eden sınıf"""
    
    def __init__(self, track_id: str, initial_detection: Dict):
        """
        Takip nesnesi başlatıcı
        
        Args:
            track_id: Benzersiz takip kimliği
            initial_detection: İlk tespit bilgileri
        """
        # Takip ID'si
        self.track_id = track_id
        # Sınırlayıcı kutu geçmişi
        self.bbox_history = [initial_detection['bbox']]
        # Merkez nokta geçmişi
        self.center_history = [initial_detection['center']]
        # Güven skoru geçmişi
        self.confidence_history = [initial_detection['confidence']]
        # Sınıf bilgisi
        self.class_name = initial_detection['class']
        # Kayıp frame sayacı
        self.lost_frames = 0
        # Toplam görülme sayısı
        self.age = 1
        # Son güncelleme zamanı (frame numarası)
        self.last_update = 0
        
    def update(self, detection: Dict):
        """Takibi yeni tespit ile güncelle"""
        # Geçmişlere yeni bilgileri ekle
        self.bbox_history.append(detection['bbox'])
        self.center_history.append(detection['center'])
        self.confidence_history.append(detection['confidence'])
        # Kayıp frame sayacını sıfırla
        self.lost_frames = 0
        # Yaşı artır
        self.age += 1
        
    def mark_lost(self):
        """Takibi kayıp olarak işaretle"""
        # Kayıp frame sayacını artır
        self.lost_frames += 1
        
    def predict_next_position(self) -> Tuple[int, int]:
        """Basit doğrusal tahmin ile sonraki pozisyonu tahmin et"""
        # En az 2 geçmiş pozisyon gerekli
        if len(self.center_history) < 2:
            return self.center_history[-1]
            
        # Son iki merkez noktasını al
        prev_center = self.center_history[-2]
        curr_center = self.center_history[-1]
        
        # Hız vektörünü hesapla
        vx = curr_center[0] - prev_center[0]
        vy = curr_center[1] - prev_center[1]
        
        # Sonraki pozisyonu tahmin et
        next_x = curr_center[0] + vx
        next_y = curr_center[1] + vy
        
        return (int(next_x), int(next_y))
    
    def get_latest_bbox(self) -> List[int]:
        """En son sınırlayıcı kutuyu döndür"""
        return self.bbox_history[-1]
    
    def get_latest_center(self) -> Tuple[int, int]:
        """En son merkez noktasını döndür"""
        return self.center_history[-1]
    
    def get_average_confidence(self) -> float:
        """Ortalama güven skorunu hesapla"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)