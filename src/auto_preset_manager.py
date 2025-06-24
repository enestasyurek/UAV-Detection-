"""
Otomatik preset yönetimi için akıllı sistem
Drone tespiti yapıldığında otomatik olarak en uygun ayarlara geçer
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class DroneDetectionInfo:
    """Drone tespit bilgileri"""
    bbox_area: float  # Bounding box alanı
    confidence: float  # Tespit güveni
    frame_resolution: Tuple[int, int]  # Frame çözünürlüğü
    detection_time: float  # Tespit zamanı
    estimated_distance: str  # Tahmini mesafe ("near", "medium", "far")


class AutoPresetManager:
    """Otomatik preset yönetici sınıfı"""
    
    def __init__(self, config_manager, update_callback=None):
        """
        Args:
            config_manager: ConfigManager instance
            update_callback: UI güncelleme callback fonksiyonu
        """
        self.config_manager = config_manager
        self.update_callback = update_callback
        self.tracking_state_manager = None  # Will be set later
        
        # Durum değişkenleri
        self.current_auto_preset = None
        self.last_detection_time = 0
        self.detection_history = []
        self.auto_mode_enabled = True
        self.last_preset_change_time = 0
        self.preset_change_cooldown = 3.0  # 3 saniye cooldown
        
        # Çözünürlük eşikleri
        self.LOW_RES_THRESHOLD = (640, 480)
        self.MEDIUM_RES_THRESHOLD = (1280, 720)
        
        # Drone boyut eşikleri (bbox alanının frame alanına oranı)
        self.NEAR_DISTANCE_THRESHOLD = 0.05  # %5'ten büyük
        self.FAR_DISTANCE_THRESHOLD = 0.005  # %0.5'ten küçük
        
        # İlk tespit için optimize edilmiş ayarlar
        self.first_detection_optimized = False
        
        logger.info("AutoPresetManager başlatıldı")
    
    def analyze_frame_resolution(self, frame_shape: Tuple[int, int, int]) -> str:
        """Frame çözünürlüğünü analiz et"""
        height, width = frame_shape[:2]
        total_pixels = height * width
        
        if width <= self.LOW_RES_THRESHOLD[0] or height <= self.LOW_RES_THRESHOLD[1]:
            return "low"
        elif width <= self.MEDIUM_RES_THRESHOLD[0] or height <= self.MEDIUM_RES_THRESHOLD[1]:
            return "medium"
        else:
            return "high"
    
    def estimate_drone_distance(self, bbox: List[float], frame_shape: Tuple[int, int]) -> str:
        """Drone mesafesini tahmin et"""
        # Bounding box alanını hesapla
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Frame alanını hesapla
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Oran hesapla
        area_ratio = bbox_area / frame_area
        
        if area_ratio > self.NEAR_DISTANCE_THRESHOLD:
            return "near"
        elif area_ratio < self.FAR_DISTANCE_THRESHOLD:
            return "far"
        else:
            return "medium"
    
    def get_optimal_preset(self, resolution: str, distance: str, confidence: float, 
                          lighting_conditions: str = "auto") -> str:
        """En uygun preset'i belirle"""
        
        # İlk tespit durumu - agresif ayarlar
        if not self.first_detection_optimized:
            logger.info("İlk drone tespiti! Agresif ayarlara geçiliyor.")
            self.first_detection_optimized = True
            
            if resolution == "low":
                return "low_res_far"  # Düşük çözünürlük için en hassas
            else:
                return "balanced"  # Diğer durumlar için dengeli başla
        
        # Düşük çözünürlük durumu - HER ZAMAN optimize et
        if resolution == "low":
            if distance == "far":
                return "low_res_far"  # En hassas ayarlar
            else:
                # Düşük çözünürlükte yakın drone - hala hassas ayarlar
                return "low_res_far"  # Düşük çözünürlük her zaman hassas ayar kullanır
        
        # Yüksek çözünürlük durumu
        elif resolution == "high":
            if distance == "near":
                return "high_res_near"
            elif distance == "far":
                return "balanced"  # Yüksek çözünürlükte uzak için dengeli
            else:
                return "day_vision" if lighting_conditions != "night" else "night_vision"
        
        # Orta çözünürlük durumu
        else:
            if distance == "far":
                return "balanced"
            elif distance == "near" and confidence > 0.7:
                return "speed_optimized"  # Yakın ve net görünüyorsa hız odaklı
            else:
                return "balanced"
    
    def set_tracking_state_manager(self, tracking_state_manager):
        """Tracking state manager'ı ayarla"""
        self.tracking_state_manager = tracking_state_manager
        logger.info("Tracking state manager bağlandı")
    
    def process_detection(self, detections: List[Dict], frame_shape: Tuple[int, int, int], 
                         current_time: float = None) -> Optional[str]:
        """
        Tespit sonuçlarını işle ve gerekirse preset değiştir
        
        Returns:
            Değiştirilecek preset adı veya None
        """
        if not self.auto_mode_enabled:
            return None
        
        if current_time is None:
            current_time = time.time()
        
        # Cooldown kontrolü
        if current_time - self.last_preset_change_time < self.preset_change_cooldown:
            return None
        
        # Drone tespiti var mı kontrol et
        drone_detections = [d for d in detections if d.get('class', '').lower() in 
                           ['drone', 'uav', 'quadcopter']]
        
        if not drone_detections:
            return None
        
        # İlk drone tespiti!
        if not self.first_detection_optimized:
            logger.info("İLK DRONE TESPİTİ! Otomatik optimizasyon başlatılıyor.")
        
        # Çözünürlük analizi
        resolution = self.analyze_frame_resolution(frame_shape)
        
        # En yüksek güvenli drone'u seç
        best_drone = max(drone_detections, key=lambda d: d.get('confidence', 0))
        
        # Mesafe tahmini
        if 'bbox' in best_drone:
            distance = self.estimate_drone_distance(best_drone['bbox'], frame_shape[:2])
        else:
            distance = "medium"  # Varsayılan
        
        # Tespit bilgisini kaydet
        detection_info = DroneDetectionInfo(
            bbox_area=(best_drone['bbox'][2] - best_drone['bbox'][0]) * 
                     (best_drone['bbox'][3] - best_drone['bbox'][1]) if 'bbox' in best_drone else 0,
            confidence=best_drone.get('confidence', 0),
            frame_resolution=(frame_shape[1], frame_shape[0]),
            detection_time=current_time,
            estimated_distance=distance
        )
        
        self.detection_history.append(detection_info)
        
        # Geçmiş temizleme (son 10 tespit)
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        # Aydınlatma koşullarını belirle (basit yaklaşım)
        lighting = "auto"  # Gelecekte frame analizi ile belirlenebilir
        
        # Tracking state manager varsa onun önerisini kullan
        if self.tracking_state_manager:
            tracking_info = self.tracking_state_manager.process_detections(
                detections, frame_shape, current_time
            )
            
            # Tracking state'e göre preset önerisi varsa kullan
            if tracking_info.get('recommended_preset'):
                optimal_preset = tracking_info['recommended_preset']
                logger.info(f"Tracking state önerisi kullanılıyor: {optimal_preset}")
            else:
                # Yoksa normal algoritma ile belirle
                optimal_preset = self.get_optimal_preset(
                    resolution, distance, best_drone.get('confidence', 0), lighting
                )
        else:
            # Tracking state manager yoksa normal algoritma
            optimal_preset = self.get_optimal_preset(
                resolution, distance, best_drone.get('confidence', 0), lighting
            )
        
        # Preset değişikliği gerekli mi?
        if optimal_preset != self.current_auto_preset:
            logger.info(f"Otomatik preset değişimi: {self.current_auto_preset} -> {optimal_preset}")
            logger.info(f"Sebep: Çözünürlük={resolution}, Mesafe={distance}, "
                       f"Güven={best_drone.get('confidence', 0):.2f}")
            
            self.current_auto_preset = optimal_preset
            self.last_preset_change_time = current_time
            
            # Preset ismini UI formatına çevir
            preset_name_mapping = {
                "low_res_far": "Düşük Çözünürlük - Uzak Mesafe",
                "high_res_near": "Yüksek Çözünürlük - Yakın Mesafe",
                "night_vision": "Gece Görüş",
                "day_vision": "Gündüz Görüş", 
                "balanced": "Dengeli",
                "speed_optimized": "Hız Odaklı"
            }
            
            return preset_name_mapping.get(optimal_preset, optimal_preset)
        
        return None
    
    def enable_auto_mode(self, enabled: bool = True):
        """Otomatik modu aç/kapa"""
        self.auto_mode_enabled = enabled
        logger.info(f"Otomatik preset modu: {'Açık' if enabled else 'Kapalı'}")
    
    def reset(self):
        """Durumu sıfırla"""
        self.current_auto_preset = None
        self.last_detection_time = 0
        self.detection_history.clear()
        self.first_detection_optimized = False
        self.last_preset_change_time = 0
        logger.info("AutoPresetManager sıfırlandı")
    
    def get_status(self) -> Dict:
        """Mevcut durumu al"""
        return {
            'enabled': self.auto_mode_enabled,
            'current_preset': self.current_auto_preset,
            'first_detection_optimized': self.first_detection_optimized,
            'detection_count': len(self.detection_history),
            'last_detection': self.detection_history[-1] if self.detection_history else None
        }