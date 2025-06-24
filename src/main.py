# Sistem modüllerini içe aktar
import sys
import os
# Loglama için logging modülünü içe aktar
import logging
# Argüman ayrıştırma için argparse modülünü içe aktar
import argparse
# Tkinter GUI kütüphanesini içe aktar
import tkinter as tk

# Proje yolunu Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# src dizinini de ekle
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# UI modülünü içe aktar
from ui import DroneTrackingApp


def setup_logging(log_level: str = "INFO"):
    """
    Loglama yapılandırması
    
    Args:
        log_level: Log seviyesi
    """
    # Log formatı
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Temel yapılandırma
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler("drone_tracking.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # OpenCV loglarını sustur
    logging.getLogger("cv2").setLevel(logging.WARNING)


def main():
    """Ana fonksiyon"""
    # Argüman ayrıştırıcı oluştur
    parser = argparse.ArgumentParser(
        description="Drone Tespit ve Takip Sistemi"
    )
    
    # Argümanları tanımla
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi (varsayılan: INFO)"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Özel YOLO model yolu (opsiyonel)"
    )
    
    parser.add_argument(
        "--video",
        default=None,
        help="Video dosyası yolu (opsiyonel)"
    )
    
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Kamerayı otomatik aç"
    )
    
    # Argümanları ayrıştır
    args = parser.parse_args()
    
    # Loglama yapılandırması
    setup_logging(args.log_level)
    
    # Logger oluştur
    logger = logging.getLogger(__name__)
    logger.info("Drone Tespit ve Takip Sistemi başlatılıyor...")
    
    # Tkinter ana penceresi oluştur
    root = tk.Tk()
    
    # Uygulama örneği oluştur
    app = DroneTrackingApp(root)
    
    # Komut satırı argümanlarını işle
    if args.model:
        # Özel model yükle
        try:
            # FastDroneDetector özel model yüklemesini desteklemiyor, 
            # bu yüzden uyarı ver
            logger.warning("FastDroneDetector kullanılıyor, özel model parametresi göz ardı edildi")
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            
    if args.video:
        # Video dosyası ayarla
        app.video_source = args.video
        app.source_label.config(text=f"Kaynak: {os.path.basename(args.video)}")
        app.start_button.config(state="normal")
        logger.info(f"Video dosyası ayarlandı: {args.video}")
        
    elif args.camera:
        # Kamerayı aç
        app._open_camera()
        logger.info("Kamera açıldı")
    
    # Uygulamayı çalıştır
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Uygulama kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"Uygulama hatası: {e}", exc_info=True)
    finally:
        logger.info("Uygulama kapatıldı")


if __name__ == "__main__":
    main()