# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Ultralytics kütüphanesinden YOLO modelini içe aktar
from ultralytics import YOLO
# PyTorch kütüphanesini GPU desteği için içe aktar
import torch
import torch.nn as nn
import torch.nn.functional as F
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Tuple, Dict, Optional
# Loglama için logging modülünü içe aktar
import logging
# Time modülünü içe aktar
import time
# Math modülünü içe aktar
import math
# Drone classifier'ı içe aktar
from .drone_classifier import DroneClassifier
# Background subtractor'ı içe aktar
from .background_subtractor import BackgroundSubtractor
# Advanced aerial classifier'ı içe aktar
from .advanced_aerial_classifier import AerialObjectClassifier

# Logger oluştur
logger = logging.getLogger(__name__)


class EdgeGuidanceSaliency(nn.Module):
    """Edge-Guided Saliency modülü - küçük nesneler için"""
    
    def __init__(self):
        super().__init__()
        # Sobel edge detection kernels
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Saliency enhancement layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        """Edge-guided saliency map oluştur"""
        # Gri tonlamaya çevir
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            gray = x
            
        # Edge detection
        edges_x = F.conv2d(gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Saliency enhancement
        saliency = F.relu(self.conv1(edges))
        saliency = F.relu(self.conv2(saliency))
        saliency = torch.sigmoid(self.conv3(saliency))
        
        return saliency


class EDGSYOLOv8Detector:
    """EDGS-YOLOv8: Edge-Guided Saliency ile geliştirilmiş drone detector"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.35,  # Optimize edilmiş eşik - %35
                 use_gpu: bool = True,
                 enable_edgs: bool = True,
                 multi_scale: bool = False):
        """
        EDGS-YOLOv8 Detector
        
        Args:
            model_path: Model dosyası yolu
            confidence_threshold: Güven eşiği
            use_gpu: GPU kullan
            enable_edgs: Edge-guided saliency aktif
            multi_scale: Çoklu ölçek tespiti
        """
        self.confidence_threshold = confidence_threshold
        self.enable_edgs = enable_edgs
        self.multi_scale = multi_scale
        
        # Device seçimi
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            
        # YOLOv8 model - nano hızlı ama EDGS ile güçlendirilmiş
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Hız için nano model kullan, EDGS zaten hassasiyeti artıracak
            self.model = YOLO('yolov8n.pt')  # Nano model - hızlı
            
        self.model.to(self.device)
        
        # FP16 optimization
        if self.device.type == 'cuda':
            self.model.model.half()
            
        # Edge-Guided Saliency modülü
        if self.enable_edgs:
            self.edgs = EdgeGuidanceSaliency().to(self.device)
            self.edgs.eval()
            
        # Drone sınıflandırıcı
        self.drone_classifier = DroneClassifier()
        
        # Gelişmiş hava aracı sınıflandırıcı (uçak, helikopter, kuş, drone)
        self.aerial_classifier = AerialObjectClassifier()
        
        # Arka plan çıkarıcı
        self.use_background_subtraction = True  # Arka plan çıkarma aktif
        self.bg_subtractor = BackgroundSubtractor()
        
        # Çoklu ölçek parametreleri - uzak drone'lar için genişletilmiş
        self.scales = [0.5, 0.75, 1.0] if multi_scale else [1.0]
        
        # Performans optimizasyonu
        self.frame_count = 0
        self.saliency_interval = 3  # Her 3 frame'de bir saliency hesapla
        
        # Gece/Gündüz adaptasyonu
        self.night_mode = False
        self.adaptive_threshold = True
        
        logger.info(f"EDGS-YOLOv8 initialized on {self.device}")
        
    def _init_drone_classifier(self):
        """Drone sınıflandırıcı - sadece drone'ları ayırt et"""
        # Basit CNN classifier
        classifier = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)  # drone / not_drone
        )
        classifier.to(self.device)
        classifier.eval()
        return classifier
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        EDGS-YOLOv8 ile drone tespiti
        
        Args:
            frame: BGR görüntü
            
        Returns:
            Tespit listesi
        """
        # Frame count güncelle
        self.frame_count += 1
        
        # Gece/Gündüz tespiti
        self._detect_lighting_condition(frame)
        
        # Arka plan çıkarma ve hareket algılama
        if self.use_background_subtraction:
            try:
                motion_info = self.bg_subtractor.process_frame(frame)
                motion_mask = motion_info['motion_mask']
                motion_regions = motion_info['motion_regions']
                enhanced_frame = motion_info['enhanced_frame']
                
                # Boyut kontrolü - motion_mask frame ile aynı boyutta olmalı
                if motion_mask.shape[:2] != frame.shape[:2]:
                    motion_mask = cv2.resize(motion_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                if enhanced_frame.shape != frame.shape:
                    enhanced_frame = cv2.resize(enhanced_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
            except Exception as e:
                logger.warning(f"Background subtraction error: {e}")
                motion_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                motion_regions = []
                enhanced_frame = frame
        else:
            motion_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            motion_regions = []
            enhanced_frame = frame
        
        # Ön işleme - hareket bölgelerinde güçlendirilmiş görüntü kullan
        processed_frame = self._preprocess_frame_with_motion(frame, enhanced_frame, motion_mask)
        
        # Çoklu ölçek tespiti
        all_detections = []
        
        for scale in self.scales:
            # Ölçeklenmiş görüntü
            if scale != 1.0:
                h, w = processed_frame.shape[:2]
                scaled_frame = cv2.resize(processed_frame, 
                                        (int(w * scale), int(h * scale)),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                scaled_frame = processed_frame
                
            # Edge-guided saliency (PERFORMANS: Belirli aralıklarla)
            if self.enable_edgs and (self.frame_count % self.saliency_interval == 0):
                saliency_map = self._compute_saliency(scaled_frame)
                # Saliency ile görüntüyü güçlendir
                scaled_frame = self._apply_saliency(scaled_frame, saliency_map)
                
            # YOLO tespiti - UZAK DRONE'LAR İÇİN DÜŞÜK EŞİK
            results = self.model(
                scaled_frame,
                conf=self._get_adaptive_threshold() * 0.3,  # Çok düşük eşik
                iou=0.2,  # Düşük IoU (uzak nesneler için)
                half=True if self.device.type == 'cuda' else False,
                verbose=False,
                agnostic_nms=True  # Sınıf farkı gözetmez
            )
            
            # Sonuçları işle
            scale_detections = self._process_results(results, frame, scale)
            all_detections.extend(scale_detections)
            
        # NMS uygula
        filtered_detections = self._nms_filtering(all_detections)
        
        # Drone sınıflandırma
        final_detections = self._classify_drones(frame, filtered_detections)
        
        # Hareket bölgelerini de ekle (eğer YOLO kaçırdıysa)
        if hasattr(self, 'bg_subtractor'):
            motion_info = self.bg_subtractor.motion_masks
            if len(motion_info) > 0:
                # Son frame'deki hareket bölgelerini kontrol et
                motion_detections = self._add_motion_based_detections(frame)
                final_detections.extend(motion_detections)
        
        return final_detections
        
    def _detect_lighting_condition(self, frame: np.ndarray):
        """Gece/Gündüz durumu tespiti"""
        # HSV'ye çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # V (Value) kanalının ortalaması
        avg_brightness = np.mean(hsv[:, :, 2])
        
        # Gece modu tespiti
        self.night_mode = avg_brightness < 80
        
        if self.night_mode:
            logger.debug("Night mode detected")
            
    def _get_adaptive_threshold(self) -> float:
        """Adaptif güven eşiği - geliştirilmiş"""
        if not self.adaptive_threshold:
            return self.confidence_threshold
            
        base_threshold = self.confidence_threshold
        
        # Gece modunda daha düşük eşik
        if self.night_mode:
            base_threshold *= 0.8  # %20 azalt
            
        # Hareket varsa daha düşük eşik
        if hasattr(self, 'has_significant_motion') and self.has_significant_motion:
            base_threshold *= 0.9  # %10 azalt
            
        # Minimum ve maksimum sınırlar
        return max(0.2, min(0.5, base_threshold))
            
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Görüntü ön işleme - HİÇBİR ŞEY YAPMA (PERFORMANS)"""
        # Performans için direkt dön
        return frame
        
    def _preprocess_frame_with_motion(self, original: np.ndarray, enhanced: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Hareket bilgisiyle görüntü ön işleme"""
        # Boyut kontrolü - tüm arrays aynı boyutta olmalı
        h, w = original.shape[:2]
        
        # Enhanced frame boyutunu kontrol et
        if enhanced.shape[:2] != (h, w):
            enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # Motion mask boyutunu kontrol et
        if motion_mask.shape[:2] != (h, w):
            motion_mask = cv2.resize(motion_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Motion mask dtype kontrolü
        if motion_mask.dtype != np.uint8:
            motion_mask = motion_mask.astype(np.uint8)
            
        # Hareket olan bölgelerde iyileştirilmiş görüntüyü kullan
        if motion_mask.any():
            # Motion mask'ı 3 kanala genişlet
            motion_mask_3ch = cv2.merge([motion_mask, motion_mask, motion_mask])
            
            # Hareket bölgelerinde enhanced, diğerlerinde original kullan
            result = np.where(motion_mask_3ch > 0, enhanced, original)
            
            # Result'ı uint8'e cast et
            result = result.astype(np.uint8)
            
            # Edge detection ekle
            edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 100, 200)
            edges_colored = cv2.merge([edges // 4, edges // 4, edges // 4])
            result = cv2.add(result, edges_colored)
            
            return result.astype(np.uint8)
        else:
            # Hareket yoksa sadece kontrast iyileştirme
            return cv2.convertScaleAbs(original, alpha=1.2, beta=10)
        
    def _compute_saliency(self, frame: np.ndarray) -> np.ndarray:
        """Edge-guided saliency hesapla"""
        # Tensor'a çevir
        img_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device) / 255.0
        
        # Saliency map
        with torch.no_grad():
            saliency = self.edgs(img_tensor)
            
        # Numpy'a çevir
        saliency_map = saliency.squeeze().cpu().numpy()
        
        # Normalize
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map
        
    def _apply_saliency(self, frame: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
        """Saliency map uygula - BASİTLEŞTİRİLMİŞ"""
        # Performans için minimal işlem
        if not self.enable_edgs:
            return frame
            
        # Basit saliency boost
        h, w = frame.shape[:2]
        saliency_resized = cv2.resize(saliency_map, (w, h))
        saliency_3d = np.stack([saliency_resized] * 3, axis=-1)
        
        # Saliency'ye göre parlaklık artır
        enhanced = frame.astype(np.float32)
        enhanced = enhanced * (1 + saliency_3d * 0.5)  # %50'ye kadar boost
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    def _process_results(self, results, original_frame: np.ndarray, scale: float) -> List[Dict]:
        """YOLO sonuçlarını işle"""
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Orijinal ölçeğe dönüştür
                x1, x2 = int(x1 / scale), int(x2 / scale)
                y1, y2 = int(y1 / scale), int(y2 / scale)
                
                # Sınırları kontrol et
                h, w = original_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Alan hesapla
                area = (x2 - x1) * (y2 - y1)
                
                # Temel filtreleme - ÇOK KÜÇÜK VEYA ÇOK BÜYÜK İSE ATLA
                if area < 4 or area > (w * h * 0.8):  # Min 2x2 piksel, max %80 ekran
                    continue
                    
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'area': area,
                    'scale': scale,
                    'aspect_ratio': (x2 - x1) / (y2 - y1) if y2 > y1 else 1.0
                }
                detections.append(detection)
                
        return detections
        
    def _nms_filtering(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """Non-Maximum Suppression"""
        if not detections:
            return []
            
        # Confidence'a göre sırala
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # En yüksek confidence'lı olanı al
            best = detections.pop(0)
            keep.append(best)
            
            # Geri kalanlarla IoU hesapla
            detections = [d for d in detections 
                         if self._calculate_iou(best['bbox'], d['bbox']) < iou_threshold]
                         
        return keep
        
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """IoU hesapla"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _classify_drones(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Gelişmiş hava aracı sınıflandırması - sadece drone'ları döndür"""
        drone_detections = []
        filtered_objects = {
            'airplane': [],
            'helicopter': [],
            'bird': [],
            'unknown': []
        }
        h, w = frame.shape[:2]
        
        for det in detections:
            # Hareket kontrolü - ÖNEMLİ!
            is_in_motion = self._check_motion_at_location(det['bbox'])
            
            # İlk basit filtreler
            aspect_ratio = det['aspect_ratio']
            area = det['area']
            
            # Çok küçük veya çok büyük nesneleri atla
            if area < 10 or area > (h * w * 0.5):
                continue
            
            # Track ID oluştur
            track_id = det.get('track_id', str(id(det)))
            
            # Gelişmiş hava aracı sınıflandırıcı ile analiz
            aerial_class, aerial_confidence, aerial_features = self.aerial_classifier.classify(
                frame, det, track_id
            )
            
            # Detay bilgileri ekle
            det['aerial_class'] = aerial_class
            det['aerial_confidence'] = aerial_confidence
            det['aerial_features'] = aerial_features
            
            # Sınıflara göre ayır
            if aerial_class == 'drone':
                # Ek drone doğrulaması (eski classifier ile)
                is_drone, drone_score = self.drone_classifier.classify_drone(frame, det)
                
                # Kombine güven skoru
                combined_confidence = aerial_confidence * 0.7 + drone_score * 0.3
                
                # Hareket bonus'u
                if is_in_motion:
                    combined_confidence = min(combined_confidence * 1.2, 1.0)
                
                # Gece modunda drone tespiti güçlendirme
                if self.night_mode:
                    combined_confidence = min(combined_confidence * 1.1, 1.0)
                
                # Drone olarak kabul et
                det['class'] = 'drone'
                det['drone_confidence'] = combined_confidence
                det['drone_score'] = drone_score
                det['is_in_motion'] = is_in_motion
                
                # Güven eşiğini kontrol et - UZAK DRONE'LAR İÇİN DÜŞÜK
                if combined_confidence > 0.15:  # Düşük eşik
                    drone_detections.append(det)
                    
                    # Debug log
                    logger.debug(f"Drone detected: confidence={combined_confidence:.2f}, "
                               f"aerial_conf={aerial_confidence:.2f}, "
                               f"motion={is_in_motion}, "
                               f"features={aerial_features}")
                    
            elif aerial_class == 'airplane':
                det['class'] = 'airplane'
                filtered_objects['airplane'].append(det)
                logger.debug(f"Airplane filtered out: confidence={aerial_confidence:.2f}")
                
            elif aerial_class == 'helicopter':
                det['class'] = 'helicopter'
                filtered_objects['helicopter'].append(det)
                logger.debug(f"Helicopter filtered out: confidence={aerial_confidence:.2f}")
                
            elif aerial_class == 'bird':
                det['class'] = 'bird'
                filtered_objects['bird'].append(det)
                logger.debug(f"Bird filtered out: confidence={aerial_confidence:.2f}")
                
            else:
                # Belirsiz nesne - düşük güvenle drone olabilir
                if is_in_motion and aerial_confidence < 0.5:
                    # Hareket varsa ve güven düşükse muhtemelen drone
                    det['class'] = 'drone'
                    det['drone_confidence'] = 0.4
                    det['uncertain'] = True
                    drone_detections.append(det)
                else:
                    det['class'] = 'unknown'
                    filtered_objects['unknown'].append(det)
                    
        # Detections'ı sakla (hareket analizi için)
        self.prev_detections = drone_detections
        
        # Frame sayısını güncelle
        self.drone_classifier.update_frame_count()
        
        # Aktif track ID'leri al ve temizlik yap
        all_detections = drone_detections + filtered_objects['airplane'] + \
                        filtered_objects['helicopter'] + filtered_objects['bird']
        active_track_ids = [d.get('track_id', str(id(d))) for d in all_detections if 'track_id' in d]
        self.aerial_classifier.cleanup_old_tracks(active_track_ids)
        
        # Filtreleme istatistikleri
        if any(filtered_objects.values()):
            logger.info(f"Filtered out: {len(filtered_objects['airplane'])} airplanes, "
                       f"{len(filtered_objects['helicopter'])} helicopters, "
                       f"{len(filtered_objects['bird'])} birds, "
                       f"{len(filtered_objects['unknown'])} unknowns")
        
        # Gece modu parametrelerini uygula
        if self.night_mode:
            night_params = self.aerial_classifier.get_night_mode_params()
            logger.debug(f"Night mode active: {night_params}")
        
        return drone_detections  # Sadece drone'ları döndür
        
    def _analyze_motion(self, detection: Dict) -> float:
        """Hareket paternini analiz et"""
        if not hasattr(self, 'prev_detections') or not self.prev_detections:
            return 0.0
            
        # En yakın önceki detection'ı bul
        min_dist = float('inf')
        for prev_det in self.prev_detections:
            dist = math.sqrt(
                (detection['center'][0] - prev_det['center'][0])**2 + 
                (detection['center'][1] - prev_det['center'][1])**2
            )
            min_dist = min(min_dist, dist)
            
        # Drone hareketi: sabit, yavaş veya orta hızlı
        if 2 < min_dist < 50:  # Makul hareket
            return 0.8
        elif min_dist < 2:  # Çok yavaş/hover
            return 0.6
        else:  # Çok hızlı veya yeni nesne
            return 0.3
        
    def detect_multi_range(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """Yakın, orta ve uzak mesafe drone tespiti"""
        detections = self.detect(frame)
        
        # Mesafeye göre sınıflandır (boyuta göre)
        h, w = frame.shape[:2]
        frame_area = h * w
        
        ranges = {
            'near': [],    # Büyük (yakın)
            'medium': [],  # Orta
            'far': []      # Küçük (uzak)
        }
        
        for det in detections:
            area_ratio = det['area'] / frame_area
            
            if area_ratio > 0.05:  # %5'ten büyük
                ranges['near'].append(det)
            elif area_ratio > 0.005:  # %0.5 - %5
                ranges['medium'].append(det)
            else:  # %0.5'ten küçük
                ranges['far'].append(det)
                
        return ranges
        
    def _check_motion_at_location(self, bbox: List[int]) -> bool:
        """Belirtilen konumda hareket var mı kontrol et"""
        if not hasattr(self, 'bg_subtractor'):
            return False
            
        # Son hareket maskesini al
        if len(self.bg_subtractor.motion_masks) == 0:
            return False
            
        motion_mask = self.bg_subtractor.motion_masks[-1]
        x1, y1, x2, y2 = bbox
        
        # Sınırları kontrol et
        if len(motion_mask.shape) == 3:
            h, w = motion_mask.shape[:2]
        else:
            h, w = motion_mask.shape
        x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
        x2, y2 = max(x1+1, min(x2, w)), max(y1+1, min(y2, h))
        
        # Bölgedeki hareket piksellerini say
        roi = motion_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False
            
        motion_pixels = np.sum(roi > 0)
        total_pixels = roi.size
        
        # %20'den fazla hareket varsa true
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
        return motion_ratio > 0.2
        
    def _add_motion_based_detections(self, frame: np.ndarray) -> List[Dict]:
        """Hareket tabanlı ek tespitler ekle"""
        motion_detections = []
        
        # Motion regions'ları al
        if not hasattr(self.bg_subtractor, 'motion_masks') or len(self.bg_subtractor.motion_masks) == 0:
            return motion_detections
            
        # Son motion mask
        motion_mask = self.bg_subtractor.motion_masks[-1]
        
        # Konturları bul
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Minimum alan kontrolü
            if area < 100:  # Çok küçük hareketleri atla
                continue
                
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Makul boyut kontrolü
            if w < 10 or h < 10 or w > frame.shape[1] * 0.3 or h > frame.shape[0] * 0.3:
                continue
                
            # Bu bölgede zaten tespit var mı kontrol et
            bbox = [x, y, x + w, y + h]
            already_detected = False
            
            # Mevcut tespitlerle çakışma kontrolü
            for det in self.prev_detections if hasattr(self, 'prev_detections') else []:
                iou = self._calculate_iou(bbox, det['bbox'])
                if iou > 0.3:  # %30'dan fazla çakışma varsa
                    already_detected = True
                    break
                    
            if not already_detected:
                # Yeni motion-based detection ekle
                cx = x + w // 2
                cy = y + h // 2
                
                motion_detections.append({
                    'bbox': bbox,
                    'confidence': 0.3,  # Düşük güven
                    'class': 'drone',
                    'class_id': 0,
                    'center': (cx, cy),
                    'area': area,
                    'aspect_ratio': w / h if h > 0 else 1.0,
                    'motion_based': True,  # Motion tabanlı olduğunu işaretle
                    'drone_confidence': 0.5,
                    'drone_score': 0.5
                })
                
        return motion_detections
        
    def warmup(self):
        """GPU ısınması"""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _ = self.detect(dummy)
        logger.info("EDGS-YOLOv8 warmed up")