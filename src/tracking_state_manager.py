"""
Drone takip durumu yöneticisi
Drone'u yakaladıktan sonra sürekli takip için optimize edilmiş sistem
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TrackingState(Enum):
    """Takip durumları"""
    NO_DETECTION = "no_detection"  # Hiç tespit yok
    FIRST_DETECTION = "first_detection"  # İlk tespit
    TRACKING_STABLE = "tracking_stable"  # Kararlı takip
    TRACKING_UNSTABLE = "tracking_unstable"  # Kararsız takip
    TRACKING_LOST = "tracking_lost"  # Takip kaybedildi
    TRACKING_RECOVERED = "tracking_recovered"  # Takip geri kazanıldı


@dataclass
class TrackedDrone:
    """Takip edilen drone bilgileri"""
    track_id: int
    last_bbox: List[float]
    last_confidence: float
    detection_count: int = 0
    lost_count: int = 0
    stable_count: int = 0
    first_detection_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    average_confidence: float = 0.0
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    size_history: List[float] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def update(self, bbox: List[float], confidence: float):
        """Drone bilgilerini güncelle"""
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.detection_count += 1
        self.lost_count = 0  # Tespit yapıldı, kayıp sayacını sıfırla
        self.last_update_time = time.time()
        
        # Güven ortalamasını güncelle
        self.average_confidence = ((self.average_confidence * (self.detection_count - 1) + confidence) / 
                                 self.detection_count)
        
        # Pozisyon ve boyut geçmişi
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.position_history.append((center_x, center_y))
        
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.size_history.append(size)
        
        # Geçmiş limitli tut
        if len(self.position_history) > 30:
            self.position_history.pop(0)
        if len(self.size_history) > 30:
            self.size_history.pop(0)
        
        # Hız hesapla
        if len(self.position_history) >= 2:
            dx = self.position_history[-1][0] - self.position_history[-2][0]
            dy = self.position_history[-1][1] - self.position_history[-2][1]
            self.velocity = (dx, dy)
    
    def mark_lost(self):
        """Drone'u kayıp olarak işaretle"""
        self.lost_count += 1
        self.stable_count = 0
    
    def is_stable(self) -> bool:
        """Takip kararlı mı?"""
        return self.stable_count >= 5 and self.lost_count == 0
    
    def get_tracking_quality(self) -> float:
        """Takip kalitesini hesapla (0-1 arası)"""
        if self.detection_count == 0:
            return 0.0
        
        # Faktörler
        confidence_factor = self.average_confidence
        stability_factor = min(self.stable_count / 10.0, 1.0)
        lost_penalty = max(1.0 - (self.lost_count / 5.0), 0.0)
        
        return (confidence_factor * 0.4 + stability_factor * 0.4 + lost_penalty * 0.2)


class TrackingStateManager:
    """Takip durumu yöneticisi"""
    
    def __init__(self, auto_preset_manager=None):
        self.auto_preset_manager = auto_preset_manager
        self.current_state = TrackingState.NO_DETECTION
        self.tracked_drones: Dict[int, TrackedDrone] = {}
        self.primary_track_id: Optional[int] = None
        self.state_change_callbacks = []
        
        # Takip parametreleri
        self.min_confidence_for_lock = 0.25
        self.max_lost_frames = 15  # 15 frame kayıp toleransı
        self.min_stable_frames = 5  # 5 frame kararlı takip için
        
        # Dinamik ayar parametreleri
        self.last_preset_update_time = 0
        self.preset_update_interval = 2.0  # 2 saniyede bir güncelle
        
        logger.info("TrackingStateManager başlatıldı")
    
    def process_detections(self, detections: List[Dict], frame_shape: Tuple[int, int, int], 
                           current_time: float = None) -> Dict[str, any]:
        """
        Tespit sonuçlarını işle ve takip durumunu güncelle
        
        Returns:
            Dict içinde takip durumu ve önerilen ayarlar
        """
        if current_time is None:
            current_time = time.time()
        
        # Drone tespitlerini filtrele
        drone_detections = {
            d.get('track_id', -1): d 
            for d in detections 
            if d.get('class', '').lower() in ['drone', 'uav', 'quadcopter']
        }
        
        # Mevcut takipleri güncelle
        self._update_tracked_drones(drone_detections, current_time)
        
        # Takip durumunu belirle
        new_state = self._determine_tracking_state(len(drone_detections))
        
        # Durum değişimi varsa callback'leri çağır
        if new_state != self.current_state:
            self._handle_state_change(self.current_state, new_state)
            self.current_state = new_state
        
        # Dinamik preset önerisi
        preset_recommendation = None
        if current_time - self.last_preset_update_time > self.preset_update_interval:
            preset_recommendation = self._get_dynamic_preset_recommendation(frame_shape)
            self.last_preset_update_time = current_time
        
        # Sonuçları döndür
        return {
            'state': self.current_state,
            'primary_drone': self.get_primary_drone(),
            'tracking_quality': self.get_tracking_quality(),
            'recommended_preset': preset_recommendation,
            'tracked_count': len(self.tracked_drones),
            'confidence_adjustment': self._get_confidence_adjustment()
        }
    
    def _update_tracked_drones(self, detections: Dict[int, Dict], current_time: float):
        """Takip edilen drone'ları güncelle"""
        detected_ids = set(detections.keys())
        
        # Mevcut takipleri güncelle
        for track_id in list(self.tracked_drones.keys()):
            if track_id in detected_ids:
                # Tespit edildi, güncelle
                detection = detections[track_id]
                self.tracked_drones[track_id].update(
                    detection['bbox'],
                    detection.get('confidence', 0.5)
                )
                self.tracked_drones[track_id].stable_count += 1
            else:
                # Tespit edilmedi, kayıp olarak işaretle
                self.tracked_drones[track_id].mark_lost()
                
                # Çok fazla kayıpsa sil
                if self.tracked_drones[track_id].lost_count > self.max_lost_frames:
                    del self.tracked_drones[track_id]
                    if self.primary_track_id == track_id:
                        self.primary_track_id = None
        
        # Yeni tespitleri ekle
        for track_id, detection in detections.items():
            if track_id not in self.tracked_drones and track_id != -1:
                # Yeni drone
                new_drone = TrackedDrone(
                    track_id=track_id,
                    last_bbox=detection['bbox'],
                    last_confidence=detection.get('confidence', 0.5)
                )
                new_drone.detection_count = 1
                self.tracked_drones[track_id] = new_drone
                
                # İlk drone ise primary yap
                if self.primary_track_id is None:
                    self.primary_track_id = track_id
                    logger.info(f"Primary drone seçildi: Track ID {track_id}")
    
    def _determine_tracking_state(self, detection_count: int) -> TrackingState:
        """Takip durumunu belirle"""
        if not self.tracked_drones:
            return TrackingState.NO_DETECTION
        
        if self.primary_track_id and self.primary_track_id in self.tracked_drones:
            primary_drone = self.tracked_drones[self.primary_track_id]
            
            if primary_drone.detection_count == 1:
                return TrackingState.FIRST_DETECTION
            elif primary_drone.is_stable():
                return TrackingState.TRACKING_STABLE
            elif primary_drone.lost_count > 0:
                if primary_drone.lost_count < 5:
                    return TrackingState.TRACKING_UNSTABLE
                else:
                    return TrackingState.TRACKING_LOST
            elif primary_drone.stable_count < self.min_stable_frames:
                return TrackingState.TRACKING_UNSTABLE
            else:
                return TrackingState.TRACKING_STABLE
        else:
            # Primary drone yok, yeni seç
            if self.tracked_drones:
                # En yüksek güvenli drone'u seç
                best_drone = max(self.tracked_drones.values(), 
                               key=lambda d: d.get_tracking_quality())
                self.primary_track_id = best_drone.track_id
                return TrackingState.TRACKING_RECOVERED
            return TrackingState.NO_DETECTION
    
    def _handle_state_change(self, old_state: TrackingState, new_state: TrackingState):
        """Durum değişikliğini yönet"""
        logger.info(f"Takip durumu değişti: {old_state.value} -> {new_state.value}")
        
        # Callback'leri çağır
        for callback in self.state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback hatası: {e}")
    
    def _get_dynamic_preset_recommendation(self, frame_shape: Tuple[int, int, int]) -> Optional[str]:
        """Takip durumuna göre dinamik preset önerisi"""
        if not self.auto_preset_manager:
            return None
        
        state = self.current_state
        primary_drone = self.get_primary_drone()
        
        if state == TrackingState.NO_DETECTION:
            # Tespit yok, en hassas ayarlar
            return "low_res_far"
        
        elif state == TrackingState.FIRST_DETECTION:
            # İlk tespit, agresif ayarlar
            return "balanced"
        
        elif state == TrackingState.TRACKING_STABLE:
            # Kararlı takip
            if primary_drone:
                # Boyuta göre ayarla
                bbox_size = primary_drone.size_history[-1] if primary_drone.size_history else 0
                frame_area = frame_shape[0] * frame_shape[1]
                size_ratio = bbox_size / frame_area if frame_area > 0 else 0
                
                if size_ratio > 0.05:  # Büyük/yakın
                    return "high_res_near"
                elif size_ratio < 0.005:  # Küçük/uzak
                    return "low_res_far"
                else:
                    return "balanced"
            return "balanced"
        
        elif state == TrackingState.TRACKING_UNSTABLE:
            # Kararsız takip, hassasiyeti artır
            return "low_res_far"
        
        elif state == TrackingState.TRACKING_LOST:
            # Takip kaybedildi, maksimum hassasiyet
            return "low_res_far"
        
        elif state == TrackingState.TRACKING_RECOVERED:
            # Takip geri kazanıldı
            return "balanced"
        
        return None
    
    def _get_confidence_adjustment(self) -> float:
        """Takip durumuna göre güven eşiği ayarı"""
        state = self.current_state
        
        if state == TrackingState.NO_DETECTION:
            return -0.1  # Daha düşük eşik
        elif state == TrackingState.TRACKING_STABLE:
            return 0.05  # Biraz yükselt
        elif state == TrackingState.TRACKING_UNSTABLE:
            return -0.05  # Biraz düşür
        elif state == TrackingState.TRACKING_LOST:
            return -0.15  # Çok düşür
        else:
            return 0.0
    
    def get_primary_drone(self) -> Optional[TrackedDrone]:
        """Ana takip edilen drone'u getir"""
        if self.primary_track_id and self.primary_track_id in self.tracked_drones:
            return self.tracked_drones[self.primary_track_id]
        return None
    
    def get_tracking_quality(self) -> float:
        """Genel takip kalitesini hesapla"""
        primary_drone = self.get_primary_drone()
        if primary_drone:
            return primary_drone.get_tracking_quality()
        return 0.0
    
    def add_state_change_callback(self, callback):
        """Durum değişikliği callback'i ekle"""
        self.state_change_callbacks.append(callback)
    
    def reset(self):
        """Takip durumunu sıfırla"""
        self.tracked_drones.clear()
        self.primary_track_id = None
        self.current_state = TrackingState.NO_DETECTION
        self.last_preset_update_time = 0
        logger.info("TrackingStateManager sıfırlandı")