# Sistem modüllerini içe aktar
import sys
import os
import platform
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
    Loglama yapılandırması - Platform uyumlu
    
    Args:
        log_level: Log seviyesi
    """
    # Platform algılama
    current_platform = platform.system()
    
    # Log formatı - platform spesifik
    if current_platform == "Darwin":  # macOS
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    elif current_platform == "Windows":
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:  # Linux ve diğerleri
        log_format = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    
    # Log dosyası yolu - platform uyumlu
    if current_platform == "Windows":
        log_file = os.path.join(os.path.expanduser("~"), "Documents", "drone_tracking.log")
    elif current_platform == "Darwin":  # macOS
        log_file = os.path.join(os.path.expanduser("~"), "Library", "Logs", "drone_tracking.log")
    else:  # Linux
        log_file = os.path.join(os.path.expanduser("~"), ".local", "share", "drone_tracking.log")
    
    # Log dizinini oluştur
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Temel yapılandırma
    handlers = [logging.StreamHandler()]
    
    # Dosya handler'ı ekle
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    except Exception as e:
        print(f"Log dosyası oluşturulamadı: {e}")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Platform spesifik logger ayarları
    if current_platform == "Darwin":  # macOS
        # macOS için daha az verbose logging
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # OpenCV loglarını sustur
    logging.getLogger("cv2").setLevel(logging.WARNING)
    
    # Log dosyası konumunu bildir
    logger = logging.getLogger(__name__)
    logger.info(f"Platform: {current_platform}")
    logger.info(f"Log dosyası: {log_file}")


def setup_platform_specific():
    """
    Platform spesifik ayarları yap
    """
    current_platform = platform.system()
    
    if current_platform == "Darwin":  # macOS
        # macOS için çevre değişkenleri
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        
        # macOS için Python path optimizasyonu
        if hasattr(sys, 'framework_prefix'):
            # Framework Python kullanılıyor
            pass
        
        # macOS için OpenMP ayarları (PyTorch için)
        os.environ['OMP_NUM_THREADS'] = '1'
        
    elif current_platform == "Windows":
        # Windows için console encoding
        if sys.stdout.encoding != 'utf-8':
            try:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
            except:
                pass
        
        # Windows için DPI awareness
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    
    else:  # Linux
        # Linux için X11 ayarları
        if 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':0'


def main():
    """Ana fonksiyon - Platform uyumlu"""
    # Platform spesifik ayarları yap
    setup_platform_specific()
    
    # Platform bilgisi
    current_platform = platform.system()
    platform_version = platform.release()
    
    # Argüman ayrıştırıcı oluştur
    parser = argparse.ArgumentParser(
        description=f"Drone Tespit ve Takip Sistemi ({current_platform} {platform_version})"
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
    
    # Platform kontrolleri
    logger.info(f"Platform: {current_platform} {platform_version}")
    logger.info(f"Python sürümü: {sys.version}")
    
    # Tkinter kontrolü ve platform spesifik ayarlar
    try:
        # Tkinter ana penceresi oluştur
        root = tk.Tk()
        
        # Platform spesifik Tkinter ayarları
        if current_platform == "Darwin":  # macOS
            # macOS için app metadata
            root.createcommand('tk::mac::ReopenApplication', lambda: root.deiconify())
            try:
                # macOS için bundle identifier
                root.tk.call('::tk::mac::standardAboutPanel')
            except:
                pass
        
        elif current_platform == "Windows":
            # Windows için taskbar icon
            try:
                import ctypes
                myappid = 'dronetracking.app.1.0'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass
        
        # Uygulama örneği oluştur
        app = DroneTrackingApp(root)
        
    except tk.TclError as e:
        logger.error(f"Tkinter başlatma hatası: {e}")
        if current_platform == "Linux":
            logger.error("Linux'ta GUI için X11 server gerekli. 'export DISPLAY=:0' komutunu deneyin.")
        elif current_platform == "Darwin":
            logger.error("macOS'ta Tkinter için XQuartz gerekli olabilir.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"GUI başlatma hatası: {e}")
        sys.exit(1)
    
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
    
    # Uygulamayı çalıştır - Platform spesifik hata yönetimi
    try:
        logger.info("Uygulama başlatılıyor...")
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Uygulama kullanıcı tarafından durduruldu (Ctrl+C)")
        
    except tk.TclError as e:
        logger.error(f"Tkinter hatası: {e}")
        if "can't invoke" in str(e).lower():
            logger.error("GUI bileşeni başlatılamadı. Platform uyumluluğu kontrol edin.")
            
    except ImportError as e:
        logger.error(f"Modül import hatası: {e}")
        if "cv2" in str(e):
            logger.error("OpenCV kurulu değil: pip install opencv-python")
        elif "PIL" in str(e):
            logger.error("Pillow kurulu değil: pip install pillow")
        elif "torch" in str(e):
            logger.error("PyTorch kurulu değil: pip install torch torchvision")
            
    except PermissionError as e:
        logger.error(f"İzin hatası: {e}")
        if current_platform == "Darwin":
            logger.error("macOS'ta kamera erişimi için izin gerekli (System Preferences > Security & Privacy)")
        elif current_platform == "Windows":
            logger.error("Windows'ta yönetici izni gerekli olabilir")
            
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}", exc_info=True)
        
        # Platform spesifik hata bilgileri
        if current_platform == "Darwin":
            logger.error("macOS sorun giderme: 'brew install python-tk' komutunu deneyin")
        elif current_platform == "Windows":
            logger.error("Windows sorun giderme: Visual C++ Redistributable kurulu olduğundan emin olun")
        elif current_platform == "Linux":
            logger.error("Linux sorun giderme: 'sudo apt-get install python3-tk' komutunu deneyin")
            
    finally:
        logger.info(f"Uygulama kapatıldı ({current_platform})")
        
        # Platform spesifik temizlik
        try:
            if 'app' in locals():
                if hasattr(app, 'video_capture') and app.video_capture:
                    app.video_capture.release()
        except:
            pass


if __name__ == "__main__":
    main()