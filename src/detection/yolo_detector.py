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
# Loglama işlemleri için logging modülünü içe aktar
import logging

# Bu modül için logger oluştur
logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        YOLO v11 tabanlı drone tespit sınıfı
        
        Args:
            model_path: Özel model yolu (opsiyonel)
            confidence_threshold: Tespit güven eşiği
        """
        # Tespit için güven eşiğini ayarla (drone'lar için düşük tutuyoruz)
        self.confidence_threshold = confidence_threshold
        # GPU varsa CUDA kullan, yoksa CPU kullan
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Eğer özel model yolu verilmişse onu yükle
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Varsayılan YOLO v11 nano modelini yükle (hızlı performans için)
            self.model = YOLO('yolo11n.pt')  # nano versiyonu, hızlı performans için
            
        # Modeli belirlenen cihaza (GPU/CPU) taşı
        self.model.to(self.device)
        # Model yükleme bilgisini logla
        logger.info(f"YOLO model loaded on {self.device}")
        
        # Drone tespiti için kullanılacak sınıf isimleri
        self.drone_classes = ['drone', 'uav', 'quadcopter']
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Verilen görüntüde drone tespiti yapar
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            Tespit edilen nesnelerin listesi
        """
        # YOLO modelini çalıştır ve sonuçları al
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Tespit edilen nesneleri saklamak için boş liste oluştur
        detections = []
        # Her sonuç için döngü başlat
        for result in results:
            # Eğer tespit edilen kutular varsa
            if result.boxes is not None:
                # Her kutu için döngü başlat
                for box in result.boxes:
                    # Kutunun köşe koordinatlarını al (sol üst ve sağ alt)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Tespit güven skorunu al
                    confidence = box.conf[0].cpu().numpy()
                    # Sınıf ID'sini al
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Sınıf ID'sine karşılık gelen sınıf adını al
                    class_name = self.model.names[class_id]
                    
                    # Eğer özel eğitilmiş model değilse, drone benzeri nesneleri filtrele
                    if not hasattr(self.model, 'custom_trained'):
                        # COCO veri setindeki drone benzeri sınıfları tanımla
                        drone_like_classes = ['airplane', 'bird', 'kite']
                        # Eğer tespit edilen nesne bu sınıflardan biri değilse atla
                        if class_name not in drone_like_classes:
                            continue
                    
                    # Tespit bilgilerini sözlük olarak sakla
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Sınırlayıcı kutu koordinatları
                        'confidence': float(confidence),  # Güven skoru
                        'class': class_name,  # Sınıf adı
                        'class_id': class_id,  # Sınıf ID'si
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Merkez noktası
                    }
                    # Tespit listesine ekle
                    detections.append(detection)
                    
        # Tüm tespitleri döndür
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Birden fazla görüntüde toplu tespit
        
        Args:
            frames: Görüntü listesi
            
        Returns:
            Her görüntü için tespit listesi
        """
        # YOLO modelini tüm görüntüler için çalıştır
        results = self.model(frames, conf=self.confidence_threshold)
        # Her görüntü için tespit listesi tutmak üzere boş liste oluştur
        batch_detections = []
        
        # Her sonuç için döngü başlat
        for result in results:
            # Bu görüntü için tespit listesi
            frame_detections = []
            # Eğer tespit edilen kutular varsa
            if result.boxes is not None:
                # Her kutu için döngü başlat
                for box in result.boxes:
                    # Kutunun köşe koordinatlarını al
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Güven skorunu al
                    confidence = box.conf[0].cpu().numpy()
                    # Sınıf ID'sini al
                    class_id = int(box.cls[0].cpu().numpy())
                    # Sınıf adını al
                    class_name = self.model.names[class_id]
                    
                    # Tespit bilgilerini sözlük olarak sakla
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': class_name,
                        'class_id': class_id,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    }
                    # Bu görüntünün tespit listesine ekle
                    frame_detections.append(detection)
            
            # Bu görüntünün tespitlerini toplu listeye ekle
            batch_detections.append(frame_detections)
            
        # Tüm görüntülerin tespit listelerini döndür
        return batch_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Tespitleri görüntü üzerine çizer
        
        Args:
            frame: Orijinal görüntü
            detections: Tespit listesi
            
        Returns:
            Çizimli görüntü
        """
        # Orijinal görüntünün kopyasını oluştur (orijinali bozmamak için)
        frame_copy = frame.copy()
        
        # Her tespit için döngü başlat
        for detection in detections:
            # Sınırlayıcı kutu koordinatlarını al
            x1, y1, x2, y2 = detection['bbox']
            # Güven skorunu al
            confidence = detection['confidence']
            # Sınıf adını al
            class_name = detection['class']
            
            # Sınırlayıcı kutuyu yeşil renkte çiz (BGR: 0,255,0), kalınlık 2 piksel
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Etiket metnini oluştur (sınıf adı ve güven skoru)
            label = f"{class_name}: {confidence:.2f}"
            # Etiket boyutunu hesapla
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            # Etiket için yeşil arka plan dikdörtgeni çiz
            cv2.rectangle(frame_copy, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         (0, 255, 0), -1)
            
            # Etiketi siyah metin olarak yaz
            cv2.putText(frame_copy, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 2)
            
            # Nesnenin merkez noktasını al
            cx, cy = detection['center']
            # Merkez noktasına mavi daire çiz (BGR: 255,0,0)
            cv2.circle(frame_copy, (cx, cy), 3, (255, 0, 0), -1)
            
        # Çizimli görüntüyü döndür
        return frame_copy
    
    def set_confidence_threshold(self, threshold: float):
        """Güven eşiğini güncelle"""
        # Eşik değerini 0 ile 1 arasında sınırla
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        
    def load_custom_model(self, model_path: str):
        """Özel eğitilmiş model yükle"""
        try:
            # Belirtilen yoldan YOLO modelini yükle
            self.model = YOLO(model_path)
            # Modeli belirlenen cihaza taşı
            self.model.to(self.device)
            # Modelin özel eğitilmiş olduğunu işaretle
            self.model.custom_trained = True
            # Başarılı yükleme mesajını logla
            logger.info(f"Custom model loaded from {model_path}")
        except Exception as e:
            # Hata durumunda hata mesajını logla
            logger.error(f"Error loading custom model: {e}")
            # Hatayı yukarı ilet
            raise