# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# NumPy kütüphanesini dizi işlemleri için içe aktar
import numpy as np
# Ultralytics kütüphanesinden YOLO modelini içe aktar
from ultralytics import YOLO
# PyTorch kütüphanesini GPU desteği için içe aktar
import torch
# Tip belirtimi için gerekli modülleri içe aktar
from typing import List, Tuple, Dict, Optional
# Loglama için logging modülünü içe aktar
import logging
# Threading için gerekli modülleri içe aktar
from threading import Thread, Lock
# Queue için gerekli modülleri içe aktar
from queue import Queue
# Time modülünü içe aktar
import time

# Logger oluştur
logger = logging.getLogger(__name__)


class FastDroneDetector:
    """Ultra hızlı ve hassas drone tespit sınıfı"""
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 confidence_threshold: float = 0.15,  # Çok düşük eşik
                 use_gpu: bool = True,
                 optimize_for_speed: bool = True):
        """
        Hızlı drone tespit sınıfı
        
        Args:
            model_path: Özel model yolu
            confidence_threshold: Tespit güven eşiği
            use_gpu: GPU kullan
            optimize_for_speed: Hız optimizasyonu
        """
        # Çok düşük güven eşiği
        self.confidence_threshold = confidence_threshold
        
        # GPU/CPU seçimi
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            torch.backends.cudnn.benchmark = True  # GPU optimizasyonu
        else:
            self.device = 'cpu'
            
        # Model yükleme
        if model_path:
            self.model = YOLO(model_path)
        else:
            # En hızlı nano model
            self.model = YOLO('yolov8n.pt')  # YOLOv8 nano daha hızlı
            
        self.model.to(self.device)
        
        # FP16 half precision (2x hız artışı)
        if self.device == 'cuda':
            self.model.model.half()
            
        # Hız optimizasyonları
        self.optimize_for_speed = optimize_for_speed
        if optimize_for_speed:
            self.skip_frames = 2  # Her 2 frame'de bir işle
            self.resize_factor = 0.5  # Görüntüyü küçült
            self.max_det = 50  # Maksimum tespit sayısı
        else:
            self.skip_frames = 1
            self.resize_factor = 0.75
            self.max_det = 100
            
        # Frame sayacı
        self.frame_counter = 0
        
        # Drone benzeri tüm sınıflar (geniş tutuyoruz)
        self.drone_classes = [
            'airplane', 'bird', 'kite', 'frisbee',  # COCO sınıfları
            'drone', 'uav', 'quadcopter',  # Özel sınıflar
            'helicopter', 'aircraft'  # Ek sınıflar
        ]
        
        # Asenkron işleme için
        self.detection_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.is_running = False
        
        logger.info(f"FastDroneDetector initialized on {self.device}")
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Hızlı drone tespiti
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            Tespit edilen nesnelerin listesi
        """
        # Frame atlama
        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0:
            return self.last_detections if hasattr(self, 'last_detections') else []
            
        # Hız için resize
        if self.resize_factor < 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized_frame = frame
            new_width, new_height = frame.shape[1], frame.shape[0]
            
        # Ultra hızlı tespit
        results = self.model(
            resized_frame, 
            conf=self.confidence_threshold,
            iou=0.3,  # Düşük NMS eşiği
            half=True if self.device == 'cuda' else False,
            max_det=self.max_det,
            verbose=False
        )
        
        detections = []
        scale_x = frame.shape[1] / new_width
        scale_y = frame.shape[0] / new_height
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Koordinatları al ve ölçekle
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Sınıf adını al
                    class_name = self.model.names[class_id]
                    
                    # Geniş filtreleme - her türlü uçan nesneyi drone olarak kabul et
                    if self._is_drone_like(class_name, confidence, (x2-x1)*(y2-y1)):
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': 'drone',  # Hepsini drone olarak işaretle
                            'original_class': class_name,
                            'class_id': class_id,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
                        
        # Son tespitleri sakla (frame atlama için)
        self.last_detections = detections
        return detections
        
    def _is_drone_like(self, class_name: str, confidence: float, area: float) -> bool:
        """
        Nesnenin drone benzeri olup olmadığını kontrol et
        
        Args:
            class_name: Sınıf adı
            confidence: Güven skoru
            area: Nesne alanı
            
        Returns:
            Drone benzeri mi?
        """
        # Bilinen drone sınıfları
        if class_name.lower() in [c.lower() for c in self.drone_classes]:
            return True
            
        # Küçük uçan nesneler (person, car vs. hariç)
        non_drone_classes = ['person', 'car', 'truck', 'bus', 'train', 'boat', 
                            'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant',
                            'bear', 'zebra', 'giraffe']
        
        if class_name.lower() in non_drone_classes:
            return False
            
        # Küçük ve orta boyutlu belirsiz nesneler
        if area < 50000 and confidence > 0.1:  # Küçük/orta nesne
            return True
            
        return False
        
    def detect_async(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Asenkron tespit (daha yüksek FPS için)
        
        Args:
            frame: Görüntü
            
        Returns:
            Hazırsa tespitler, değilse None
        """
        if not self.is_running:
            self.start_async_processing()
            
        # Frame'i kuyruğa ekle (bloklamadan)
        try:
            self.detection_queue.put_nowait(frame)
        except:
            pass  # Kuyruk doluysa atla
            
        # Sonuç varsa al
        try:
            return self.result_queue.get_nowait()
        except:
            return None  # Henüz sonuç yok
            
    def start_async_processing(self):
        """Asenkron işleme thread'ini başlat"""
        self.is_running = True
        self.processing_thread = Thread(target=self._async_worker, daemon=True)
        self.processing_thread.start()
        
    def _async_worker(self):
        """Arka plan işleme thread'i"""
        while self.is_running:
            try:
                frame = self.detection_queue.get(timeout=1.0)
                detections = self.detect(frame)
                
                # Sonucu kuyruğa ekle
                try:
                    self.result_queue.put_nowait(detections)
                except:
                    pass  # Kuyruk doluysa eski sonucu at
                    
            except:
                continue
                
    def stop_async_processing(self):
        """Asenkron işlemeyi durdur"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
            
    def warmup(self):
        """GPU ısınması için dummy tespit"""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.detect(dummy_frame)
        logger.info("Detector warmed up")
        
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Toplu tespit (daha verimli)
        
        Args:
            frames: Görüntü listesi
            
        Returns:
            Her görüntü için tespit listesi
        """
        if not frames:
            return []
            
        # Batch resize
        resized_frames = []
        for frame in frames:
            if self.resize_factor < 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * self.resize_factor)
                new_height = int(height * self.resize_factor)
                resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized)
            else:
                resized_frames.append(frame)
                
        # Batch inference
        results = self.model(
            resized_frames,
            conf=self.confidence_threshold,
            iou=0.3,
            half=True if self.device == 'cuda' else False,
            max_det=self.max_det,
            verbose=False
        )
        
        batch_detections = []
        for i, result in enumerate(results):
            frame_detections = []
            
            if result.boxes is not None:
                # Ölçekleme faktörleri
                scale_x = frames[i].shape[1] / resized_frames[i].shape[1]
                scale_y = frames[i].shape[0] / resized_frames[i].shape[0]
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    if self._is_drone_like(class_name, confidence, (x2-x1)*(y2-y1)):
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': 'drone',
                            'original_class': class_name,
                            'class_id': class_id,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        frame_detections.append(detection)
                        
            batch_detections.append(frame_detections)
            
        return batch_detections
        
    def get_optimized_settings(self) -> Dict:
        """Mevcut optimizasyon ayarlarını döndür"""
        return {
            'device': self.device,
            'skip_frames': self.skip_frames,
            'resize_factor': self.resize_factor,
            'confidence_threshold': self.confidence_threshold,
            'max_detections': self.max_det,
            'half_precision': self.device == 'cuda',
            'async_enabled': self.is_running
        }