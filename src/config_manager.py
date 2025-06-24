"""
Konfigürasyon yönetimi sınıfı
Hızlı ayar presetlerini yönetir ve uygular
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Konfigürasyon yönetimi sınıfı"""
    
    def __init__(self):
        """Konfigürasyon yöneticisini başlat"""
        self.presets = {}
        self.current_preset = None
        self.load_presets()
    
    def load_presets(self) -> bool:
        """Preset dosyasını yükle"""
        try:
            # Farklı yolları dene
            possible_paths = [
                # src/config_manager.py -> config/
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'quick_settings_presets.json'),
                # src/ -> ../config/
                os.path.join(os.path.dirname(__file__), '..', 'config', 'quick_settings_presets.json'),
                # Mevcut dizinden
                os.path.join('config', 'quick_settings_presets.json'),
                # Mutlak yol dene
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'quick_settings_presets.json')
            ]
            
            presets_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    presets_path = path
                    break
            
            if not presets_path:
                logger.error(f"Preset dosyası hiçbir yolda bulunamadı. Denenen yollar: {possible_paths}")
                return False
                
            logger.info(f"Preset dosyası bulundu: {presets_path}")
            with open(presets_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.presets = data.get('presets', {})
                logger.info(f"Presets yüklendi: {len(self.presets)} preset")
                return True
                
        except Exception as e:
            logger.error(f"Preset yükleme hatası: {e}")
            return False
    
    def get_preset(self, preset_key: str) -> Optional[Dict[str, Any]]:
        """Preset'i al"""
        return self.presets.get(preset_key)
    
    def get_all_presets(self) -> Dict[str, Any]:
        """Tüm presetleri al"""
        return self.presets.copy()
    
    def apply_preset_to_detector(self, detector, preset_key: str) -> bool:
        """Preset'i detector'e uygula"""
        preset = self.get_preset(preset_key)
        if not preset:
            logger.error(f"Preset bulunamadı: {preset_key}")
            return False
            
        try:
            detection_settings = preset.get('detection', {})
            
            # Confidence threshold
            if hasattr(detector, 'confidence_threshold') and 'confidence_threshold' in detection_settings:
                detector.confidence_threshold = detection_settings['confidence_threshold']
                logger.info(f"Detector confidence threshold: {detector.confidence_threshold}")
            
            # NMS threshold
            if hasattr(detector, 'nms_threshold') and 'nms_threshold' in detection_settings:
                detector.nms_threshold = detection_settings['nms_threshold']
                logger.info(f"Detector NMS threshold: {detector.nms_threshold}")
            
            # EDGS enable
            if hasattr(detector, 'enable_edgs') and 'enable_edgs' in detection_settings:
                detector.enable_edgs = detection_settings['enable_edgs']
                logger.info(f"Detector EDGS: {detector.enable_edgs}")
            
            # Multi-scale
            if hasattr(detector, 'multi_scale') and 'multi_scale' in detection_settings:
                detector.multi_scale = detection_settings['multi_scale']
                if detector.multi_scale:
                    detector.scales = [0.75, 1.0, 1.25]
                else:
                    detector.scales = [1.0]
                logger.info(f"Detector multi-scale: {detector.multi_scale}")
            
            # Max detections
            if hasattr(detector, 'max_detections') and 'max_detections' in detection_settings:
                detector.max_detections = detection_settings['max_detections']
                logger.info(f"Detector max detections: {detector.max_detections}")
            
            self.current_preset = preset_key
            logger.info(f"Detector preset uygulandı: {preset_key}")
            return True
            
        except Exception as e:
            logger.error(f"Detector preset uygulama hatası: {e}")
            return False
    
    def apply_preset_to_tracker(self, tracker, preset_key: str) -> bool:
        """Preset'i tracker'e uygula"""
        preset = self.get_preset(preset_key)
        if not preset:
            logger.error(f"Preset bulunamadı: {preset_key}")
            return False
            
        try:
            tracking_settings = preset.get('tracking', {})
            
            # Detection threshold
            if hasattr(tracker, 'det_thresh') and 'det_thresh' in tracking_settings:
                tracker.det_thresh = tracking_settings['det_thresh']
                logger.info(f"Tracker det_thresh: {tracker.det_thresh}")
            
            # Max age
            if hasattr(tracker, 'max_age') and 'max_age' in tracking_settings:
                tracker.max_age = tracking_settings['max_age']
                logger.info(f"Tracker max_age: {tracker.max_age}")
            
            # Min hits
            if hasattr(tracker, 'min_hits') and 'min_hits' in tracking_settings:
                tracker.min_hits = tracking_settings['min_hits']
                logger.info(f"Tracker min_hits: {tracker.min_hits}")
            
            # IOU threshold
            if hasattr(tracker, 'iou_threshold') and 'iou_threshold' in tracking_settings:
                tracker.iou_threshold = tracking_settings['iou_threshold']
                logger.info(f"Tracker IOU threshold: {tracker.iou_threshold}")
            
            # ByteTracker specific settings
            if hasattr(tracker, 'track_thresh') and 'det_thresh' in tracking_settings:
                tracker.track_thresh = tracking_settings['det_thresh']
                logger.info(f"ByteTracker track_thresh: {tracker.track_thresh}")
                
            if hasattr(tracker, 'track_buffer') and 'max_age' in tracking_settings:
                tracker.track_buffer = int(tracking_settings['max_age'] * 0.2)  # %20'si
                logger.info(f"ByteTracker track_buffer: {tracker.track_buffer}")
                
            if hasattr(tracker, 'match_thresh') and 'iou_threshold' in tracking_settings:
                tracker.match_thresh = tracking_settings['iou_threshold']
                logger.info(f"ByteTracker match_thresh: {tracker.match_thresh}")
            
            logger.info(f"Tracker preset uygulandı: {preset_key}")
            return True
            
        except Exception as e:
            logger.error(f"Tracker preset uygulama hatası: {e}")
            return False
    
    def apply_preset_to_preprocessor(self, preprocessor, preset_key: str) -> bool:
        """Preset'i preprocessor'e uygula"""
        preset = self.get_preset(preset_key)
        if not preset:
            logger.error(f"Preset bulunamadı: {preset_key}")
            return False
            
        try:
            preprocessing_settings = preset.get('preprocessing', {})
            
            # CLAHE enable
            if hasattr(preprocessor, 'enable_clahe') and 'enable_clahe' in preprocessing_settings:
                preprocessor.enable_clahe = preprocessing_settings['enable_clahe']
                logger.info(f"Preprocessor CLAHE: {preprocessor.enable_clahe}")
            
            # CLAHE clip limit
            if hasattr(preprocessor, 'clahe_clip_limit') and 'clahe_clip_limit' in preprocessing_settings:
                preprocessor.clahe_clip_limit = preprocessing_settings['clahe_clip_limit']
                logger.info(f"Preprocessor CLAHE clip limit: {preprocessor.clahe_clip_limit}")
            
            # CLAHE tile grid size
            if hasattr(preprocessor, 'clahe_tile_grid_size') and 'clahe_tile_grid_size' in preprocessing_settings:
                preprocessor.clahe_tile_grid_size = tuple(preprocessing_settings['clahe_tile_grid_size'])
                logger.info(f"Preprocessor CLAHE tile grid: {preprocessor.clahe_tile_grid_size}")
            
            # Gamma correction
            if hasattr(preprocessor, 'gamma_correction') and 'gamma_correction' in preprocessing_settings:
                preprocessor.gamma_correction = preprocessing_settings['gamma_correction']
                logger.info(f"Preprocessor gamma: {preprocessor.gamma_correction}")
            
            # Contrast enhancement
            if hasattr(preprocessor, 'enhance_contrast') and 'enhance_contrast' in preprocessing_settings:
                preprocessor.enhance_contrast = preprocessing_settings['enhance_contrast']
                logger.info(f"Preprocessor contrast enhancement: {preprocessor.enhance_contrast}")
            
            logger.info(f"Preprocessor preset uygulandı: {preset_key}")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessor preset uygulama hatası: {e}")
            return False
    
    def get_current_preset(self) -> Optional[str]:
        """Mevcut preset'i al"""
        return self.current_preset
    
    def get_preset_info(self, preset_key: str) -> Optional[Dict[str, str]]:
        """Preset bilgilerini al"""
        preset = self.get_preset(preset_key)
        if not preset:
            return None
            
        return {
            'name': preset.get('name', preset_key),
            'description': preset.get('description', 'Açıklama yok')
        }