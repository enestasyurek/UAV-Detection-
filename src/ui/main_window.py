# Tkinter GUI kütüphanesini içe aktar
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# OpenCV kütüphanesini görüntü işleme için içe aktar
import cv2
# PIL kütüphanesini Tkinter ile görüntü gösterimi için içe aktar
from PIL import Image, ImageTk
# Threading için gerekli modülleri içe aktar
import threading
import queue
# Zaman işlemleri için time modülünü içe aktar
import time
# NumPy kütüphanesini içe aktar
import numpy as np
# Tip belirtimi için gerekli modülleri içe aktar
from typing import Optional, Dict, List
# Proje modüllerini içe aktar
from detection import YOLODetector, FastDroneDetector, EDGSYOLOv8Detector
from tracking import ByteTracker, DeepSortTracker, OCSortTracker, QuickUAVOCSortTracker, DroneSpecificTracker
from preprocessing import ImageEnhancer, CameraMode
# Loglama için logging modülünü içe aktar
import logging

# Logger oluştur
logger = logging.getLogger(__name__)


class DroneTrackingApp:
    """Ana uygulama penceresi sınıfı"""
    
    def __init__(self, root: tk.Tk):
        """
        Uygulama başlatıcı
        
        Args:
            root: Tkinter ana penceresi
        """
        # Ana pencere referansı
        self.root = root
        self.root.title("Drone Tespit ve Takip Sistemi")
        
        # macOS optimizasyonları
        import platform
        self.is_mac = platform.system() == 'Darwin'
        
        if self.is_mac:
            # macOS için optimize edilmiş pencere boyutu ve ayarları
            self.root.geometry("1280x720")
            self.root.minsize(1024, 600)
            
            # macOS pencere özellikleri - basit yaklaşım
            self.root.configure(bg="#f5f5f5")
            
            # macOS için tema - debug ile
            style = ttk.Style()
            try:
                # Önce mevcut temaları kontrol et
                available_themes = style.theme_names()
                print(f"Mevcut temalar: {available_themes}")
                
                if 'aqua' in available_themes:
                    style.theme_use('aqua')
                    print("Aqua teması kullanılıyor")
                elif 'clam' in available_themes:
                    style.theme_use('clam')
                    print("Clam teması kullanılıyor")
                else:
                    style.theme_use('default')
                    print("Default tema kullanılıyor")
            except Exception as e:
                print(f"Tema ayarlanamadı: {e}")
                style = ttk.Style()
                style.theme_use('default')
                
            # macOS için font ayarları - güvenli yaklaşım
            try:
                import tkinter.font as tkFont
                # Sistem fontlarını kontrol et
                available_fonts = list(tkFont.families())
                print(f"Sistem fontları sayısı: {len(available_fonts)}")
                
                if "SF Pro Text" in available_fonts:
                    default_font = tkFont.nametofont("TkDefaultFont")
                    default_font.configure(family="SF Pro Text", size=13)
                    print("SF Pro Text font ayarlandı")
                elif "Helvetica" in available_fonts:
                    default_font = tkFont.nametofont("TkDefaultFont")
                    default_font.configure(family="Helvetica", size=13)
                    print("Helvetica font ayarlandı")
                    
                if "SF Mono" in available_fonts:
                    text_font = tkFont.nametofont("TkTextFont")
                    text_font.configure(family="SF Mono", size=12)
                    print("SF Mono font ayarlandı")
                elif "Monaco" in available_fonts:
                    text_font = tkFont.nametofont("TkTextFont")
                    text_font.configure(family="Monaco", size=12)
                    print("Monaco font ayarlandı")
                    
            except Exception as e:
                print(f"Font ayarlanamadı: {e}")
            
            # Retina display desteği - daha dikkatli
            try:
                # Mevcut scaling değerini al
                current_scaling = self.root.tk.call('tk', 'scaling')
                print(f"Mevcut scaling: {current_scaling}")
                
                # Eğer çok düşükse artır - ama aşırı artırma
                if current_scaling < 1.2:
                    self.root.tk.call('tk', 'scaling', 1.3)
                    print("Scaling 1.3'e ayarlandı")
            except Exception as e:
                print(f"Scaling ayarlanamadı: {e}")
            
        else:
            # Diğer platformlar için
            self.root.geometry("1400x800")
            self.root.configure(bg="#f0f0f0")
            style = ttk.Style()
            style.theme_use('clam')
        
        # Değişkenler
        self.video_source = None
        self.video_capture = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.video_fps = 30  # Varsayılan FPS
        self.process_every_n_frames = 2  # Her 2. frame'i işle (PERFORMANS)
        
        # Modüller - EDGS-YOLOv8 detector kullan
        self.detector = EDGSYOLOv8Detector(
            confidence_threshold=0.15,  # Uzak drone'lar için düşük eşik
            use_gpu=True,
            enable_edgs=True,  # Drone tespiti için AÇIK
            multi_scale=True  # Uzak drone'lar için açık
        )
        # GPU ısınması
        self.detector.warmup()
        
        self.tracker = None
        self.enhancer = ImageEnhancer()
        
        # Takip algoritması seçimi (varsayılan olarak Drone-Specific Tracker)
        self.tracker_type = tk.StringVar(value="Drone-Specific")
        self.camera_mode = tk.StringVar(value="AUTO")
        
        # İstatistikler
        self.fps = 0
        self.detection_count = 0
        self.track_count = 0
        
        # UI oluştur
        print("UI oluşturuluyor...")
        self._create_ui()
        print("UI oluşturuldu!")
        
        # Takip algoritmasını başlat
        print("Tracker başlatılıyor...")
        self._init_tracker()
        print("Tracker başlatıldı!")
        
        # macOS için ek debug
        if self.is_mac:
            self.root.update_idletasks()
            print(f"Pencere boyutu: {self.root.winfo_width()}x{self.root.winfo_height()}")
            print(f"Pencere görünür: {self.root.winfo_viewable()}")
        
    def _create_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        print("_create_ui başladı")
        
        # Ana çerçeve - macOS için optimize edilmiş
        padding = 15 if self.is_mac else 10
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)
        print(f"Main frame oluşturuldu, padding: {padding}")
        
        # macOS için grid weights ayarla
        if self.is_mac:
            main_frame.grid_columnconfigure(0, weight=0, minsize=220)  # Sol panel
            main_frame.grid_columnconfigure(1, weight=1)              # Video panel
            main_frame.grid_columnconfigure(2, weight=0, minsize=280) # Bilgi panel
            main_frame.grid_rowconfigure(0, weight=1)
            print("macOS grid weights ayarlandı")
        else:
            # Diğer platformlar için varsayılan
            main_frame.grid_columnconfigure(1, weight=1)
            main_frame.grid_rowconfigure(0, weight=1)
            print("Standart grid weights ayarlandı")
        
        # Sol panel (kontroller)
        print("Kontrol paneli oluşturuluyor...")
        self._create_control_panel(main_frame)
        print("Kontrol paneli oluşturuldu")
        
        # Orta panel (video)
        print("Video paneli oluşturuluyor...")
        self._create_video_panel(main_frame)
        print("Video paneli oluşturuldu")
        
        # Sağ panel (bilgi)
        print("Bilgi paneli oluşturuluyor...")
        self._create_info_panel(main_frame)
        print("Bilgi paneli oluşturuldu")
        
        # Alt panel (durum çubuğu)
        print("Durum çubuğu oluşturuluyor...")
        self._create_status_bar()
        print("Durum çubuğu oluşturuldu")
        
        # macOS için force update
        if self.is_mac:
            self.root.update_idletasks()
            main_frame.update_idletasks()
            print("macOS force update yapıldı")
            
        print("_create_ui tamamlandı")
        
    def _create_control_panel(self, parent):
        """Kontrol panelini oluştur"""
        # Kontrol çerçevesi - macOS için optimize edilmiş
        padding = 15 if self.is_mac else 10
        control_frame = ttk.LabelFrame(parent, text="Kontroller", padding=padding)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # macOS için column genişliği ayarla
        if self.is_mac:
            control_frame.grid_columnconfigure(0, weight=1, minsize=200)
        
        # Kaynak seçimi
        ttk.Label(control_frame, text="Video Kaynağı:").grid(row=0, column=0, sticky="w", pady=5)
        
        # Dosya seç butonu
        ttk.Button(
            control_frame, 
            text="Video Seç",
            command=self._select_video_file
        ).grid(row=1, column=0, sticky="ew", pady=5)
        
        # Kamera seç butonu
        ttk.Button(
            control_frame,
            text="Kamera Aç",
            command=self._open_camera
        ).grid(row=2, column=0, sticky="ew", pady=5)
        
        # Ayırıcı
        ttk.Separator(control_frame, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=10)
        
        # Takip algoritması seçimi
        ttk.Label(control_frame, text="Takip Algoritması:").grid(row=4, column=0, sticky="w", pady=5)
        
        tracker_combo = ttk.Combobox(
            control_frame,
            textvariable=self.tracker_type,
            values=["Drone-Specific", "Quick-UAV OC-SORT", "ByteTrack", "DeepSORT", "OC-SORT"],
            state="readonly"
        )
        tracker_combo.grid(row=5, column=0, sticky="ew", pady=5)
        tracker_combo.bind("<<ComboboxSelected>>", lambda e: self._init_tracker())
        
        # Kamera modu seçimi
        ttk.Label(control_frame, text="Kamera Modu:").grid(row=6, column=0, sticky="w", pady=5)
        
        mode_combo = ttk.Combobox(
            control_frame,
            textvariable=self.camera_mode,
            values=["AUTO", "DAY", "NIGHT"],
            state="readonly"
        )
        mode_combo.grid(row=7, column=0, sticky="ew", pady=5)
        
        # Ayırıcı
        ttk.Separator(control_frame, orient="horizontal").grid(row=8, column=0, sticky="ew", pady=10)
        
        # Kontrol butonları
        self.start_button = ttk.Button(
            control_frame,
            text="Başlat",
            command=self._start_tracking,
            state="disabled"
        )
        self.start_button.grid(row=9, column=0, sticky="ew", pady=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Durdur",
            command=self._stop_tracking,
            state="disabled"
        )
        self.stop_button.grid(row=10, column=0, sticky="ew", pady=5)
        
        # Kaydet butonu
        ttk.Button(
            control_frame,
            text="Sonuçları Kaydet",
            command=self._save_results
        ).grid(row=11, column=0, sticky="ew", pady=5)
        
        # Güven eşiği (drone'lar için çok düşük başlıyoruz)
        ttk.Label(control_frame, text="Güven Eşiği:").grid(row=12, column=0, sticky="w", pady=5)
        
        self.confidence_var = tk.DoubleVar(value=0.25)  # %25 ile başla - dengeli
        confidence_scale = ttk.Scale(
            control_frame,
            from_=0.05,  # Minimum %5 (çok düşük değerleri engelledik)
            to=0.8,  # Maximum %80
            variable=self.confidence_var,
            orient="horizontal"
        )
        confidence_scale.grid(row=13, column=0, sticky="ew", pady=5)
        
        # Güven değeri etiketi
        self.confidence_label = ttk.Label(control_frame, text="0.25")
        self.confidence_label.grid(row=14, column=0, pady=5)
        
        # Güven değişimi takibi
        confidence_scale.configure(command=self._update_confidence)
        
        # Ayırıcı
        ttk.Separator(control_frame, orient="horizontal").grid(row=15, column=0, sticky="ew", pady=10)
        
        # Gelişmiş ayarlar
        ttk.Label(control_frame, text="Gelişmiş Ayarlar:").grid(row=16, column=0, sticky="w", pady=5)
        
        # EDGS aktif/pasif
        self.enable_edgs = tk.BooleanVar(value=True)  # Drone tespiti için AÇIK
        ttk.Checkbutton(
            control_frame,
            text="Edge-Guided Saliency (Drone Tespiti)",
            variable=self.enable_edgs,
            command=self._update_detector_settings
        ).grid(row=17, column=0, sticky="w", pady=2)
        
        # Multi-scale aktif/pasif
        self.enable_multiscale = tk.BooleanVar(value=False)  # Performans için kapalı başlat
        ttk.Checkbutton(
            control_frame,
            text="Çoklu Ölçek (Uzak Drone'lar)",
            variable=self.enable_multiscale,
            command=self._update_detector_settings
        ).grid(row=18, column=0, sticky="w", pady=2)
        
    def _create_video_panel(self, parent):
        """Video panelini oluştur"""
        # Video çerçevesi
        video_frame = ttk.LabelFrame(parent, text="Video Görüntüsü", padding=10)
        video_frame.grid(row=0, column=1, sticky="nsew")
        
        # Video canvas - macOS için optimize edilmiş
        if self.is_mac:
            # macOS için Retina-ready canvas boyutları
            canvas_width, canvas_height = 720, 540
        else:
            canvas_width, canvas_height = 800, 600
            
        self.video_canvas = tk.Canvas(video_frame, width=canvas_width, height=canvas_height, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # macOS için smooth scrolling
        if self.is_mac:
            self.video_canvas.configure(highlightthickness=0)
        
        # Video kontrolleri
        controls_frame = ttk.Frame(video_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Tespit gösterimi
        self.show_detections = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_frame,
            text="Tespitleri Göster",
            variable=self.show_detections
        ).pack(side=tk.LEFT, padx=5)
        
        # Takip çizgileri
        self.show_tracks = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_frame,
            text="Takip Çizgilerini Göster",
            variable=self.show_tracks
        ).pack(side=tk.LEFT, padx=5)
        
        # İyileştirme
        self.enhance_image = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_frame,
            text="Görüntü İyileştirme",
            variable=self.enhance_image
        ).pack(side=tk.LEFT, padx=5)
        
        # Video döngüsü
        self.loop_video = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_frame,
            text="Video Döngüsü",
            variable=self.loop_video
        ).pack(side=tk.LEFT, padx=5)
        
        # Hareket maskesi göster
        self.show_motion_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls_frame,
            text="Hareket Maskesi",
            variable=self.show_motion_mask
        ).pack(side=tk.LEFT, padx=5)
        
    def _create_info_panel(self, parent):
        """Bilgi panelini oluştur"""
        # Bilgi çerçevesi
        info_frame = ttk.LabelFrame(parent, text="İstatistikler", padding=10)
        info_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        
        # Font ayarları - macOS için optimize edilmiş
        if self.is_mac:
            label_font = ("SF Pro Text", 12)
            value_font = ("SF Pro Display", 16, "bold")
        else:
            label_font = ("Arial", 10)
            value_font = ("Arial", 14, "bold")
        
        # FPS göstergesi
        ttk.Label(info_frame, text="FPS:", font=label_font).grid(row=0, column=0, sticky="w", pady=5)
        self.fps_label = ttk.Label(info_frame, text="0", font=value_font)
        self.fps_label.grid(row=0, column=1, sticky="e", pady=5)
        
        # Tespit sayısı
        ttk.Label(info_frame, text="Tespitler:", font=label_font).grid(row=1, column=0, sticky="w", pady=5)
        self.detection_label = ttk.Label(info_frame, text="0", font=value_font)
        self.detection_label.grid(row=1, column=1, sticky="e", pady=5)
        
        # Takip sayısı
        ttk.Label(info_frame, text="Aktif Takipler:", font=label_font).grid(row=2, column=0, sticky="w", pady=5)
        self.track_label = ttk.Label(info_frame, text="0", font=value_font)
        self.track_label.grid(row=2, column=1, sticky="e", pady=5)
        
        # Ayırıcı
        ttk.Separator(info_frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Tespit listesi
        ttk.Label(info_frame, text="Tespit Edilen Nesneler:").grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        # Treeview için çerçeve
        tree_frame = ttk.Frame(info_frame)
        tree_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview - macOS için optimize edilmiş
        tree_height = 8 if self.is_mac else 10
        self.detection_tree = ttk.Treeview(
            tree_frame,
            columns=("ID", "Sınıf", "Güven", "Yaş"),
            show="headings",
            height=tree_height,
            yscrollcommand=scrollbar.set
        )
        
        # macOS için alternating row colors
        if self.is_mac:
            self.detection_tree.tag_configure('oddrow', background='#f8f8f8')
            self.detection_tree.tag_configure('evenrow', background='#ffffff')
        
        # Sütun başlıkları
        self.detection_tree.heading("ID", text="Takip ID")
        self.detection_tree.heading("Sınıf", text="Sınıf")
        self.detection_tree.heading("Güven", text="Güven")
        self.detection_tree.heading("Yaş", text="Yaş")
        
        # macOS için optimize edilmiş sütun genişlikleri
        if self.is_mac:
            self.detection_tree.column("ID", width=90)
            self.detection_tree.column("Sınıf", width=100)
            self.detection_tree.column("Güven", width=80)
            self.detection_tree.column("Yaş", width=60)
        else:
            self.detection_tree.column("ID", width=80)
            self.detection_tree.column("Sınıf", width=80)
            self.detection_tree.column("Güven", width=60)
            self.detection_tree.column("Yaş", width=50)
        
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.detection_tree.yview)
        
        # Grid ağırlıkları
        info_frame.grid_rowconfigure(5, weight=1)
        
    def _create_status_bar(self):
        """Durum çubuğunu oluştur"""
        # Durum çerçevesi
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Durum etiketi
        self.status_label = ttk.Label(
            status_frame,
            text="Hazır",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        # Kaynak bilgisi
        self.source_label = ttk.Label(
            status_frame,
            text="Kaynak: Yok",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.source_label.pack(side=tk.RIGHT, padx=2, pady=2)
        
    def _init_tracker(self):
        """Seçilen takip algoritmasını başlat"""
        tracker_type = self.tracker_type.get()
        
        if tracker_type == "Drone-Specific":
            # EDGS-YOLOv8 için optimize edilmiş drone tracker
            self.tracker = DroneSpecificTracker(
                det_thresh=0.01,  # ÇOK ultra düşük eşik
                max_age=200,  # ÇOK uzun takip (200 frame)
                min_hits=1,  # Hemen onay
                iou_threshold=0.05,  # ÇOK esnek IoU
                use_visual_features=True,  # Görsel özellikler
                range_adaptive=True  # Mesafeye adaptif
            )
        elif tracker_type == "Quick-UAV OC-SORT":
            # Drone takibi için optimize edilmiş parametreler
            self.tracker = QuickUAVOCSortTracker(
                det_thresh=0.01,  # ÇOK düşük eşik
                max_age=150,  # ÇOK uzun takip süresi
                min_hits=1,  # Anında onay
                iou_threshold=0.05,  # ÇOK esnek eşleştirme
                altitude_aware=True,
                motion_prediction=True
            )
        elif tracker_type == "ByteTrack":
            self.tracker = ByteTracker(
                track_thresh=0.4,  # Drone'lar için düşürüldü
                track_buffer=40,
                match_thresh=0.7
            )
        elif tracker_type == "DeepSORT":
            self.tracker = DeepSortTracker(
                max_dist=0.3,  # Drone'lar için artırıldı
                min_confidence=0.25,
                max_age=40
            )
        elif tracker_type == "OC-SORT":
            self.tracker = OCSortTracker(
                det_thresh=0.3,
                max_age=40,
                iou_threshold=0.25
            )
            
        logger.info(f"Tracker initialized: {tracker_type}")
        
    def _select_video_file(self):
        """Video dosyası seç"""
        filename = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.video_source = filename
            self.source_label.config(text=f"Kaynak: {filename.split('/')[-1]}")
            self.start_button.config(state="normal")
            self.status_label.config(text="Video dosyası seçildi")
            
    def _open_camera(self):
        """Kamerayı aç"""
        self.video_source = 0  # Varsayılan kamera
        self.source_label.config(text="Kaynak: Kamera")
        self.start_button.config(state="normal")
        self.status_label.config(text="Kamera seçildi")
        
    def _start_tracking(self):
        """Takibi başlat"""
        if self.video_source is None:
            messagebox.showerror("Hata", "Lütfen önce bir video kaynağı seçin!")
            return
            
        # Video capture başlat
        self.video_capture = cv2.VideoCapture(self.video_source)
        
        # Buffer boyutunu azalt (gecikmeyi önle)
        if isinstance(self.video_source, int):  # Kamera
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # FPS ayarla
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            # Çözünürlüğü optimize et
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:  # Video dosyası
            # Video FPS'ini al
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.video_fps = fps
            else:
                self.video_fps = 30  # Varsayılan
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Hata", "Video kaynağı açılamadı!")
            return
            
        # Durumu güncelle
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Takip başlatıldı...")
        
        # İşlem thread'ini başlat
        self.process_thread = threading.Thread(target=self._process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Görüntüleme döngüsünü başlat
        self._update_display()
        
    def _stop_tracking(self):
        """Takibi durdur"""
        self.is_running = False
        
        # Video capture'ı kapat
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        # Takip geçmişini temizle
        if hasattr(self, 'track_histories'):
            self.track_histories.clear()
            
        # Tracker'ı sıfırla
        if self.tracker:
            self.tracker.trackers = []  # Tüm takipleri temizle
            
        # Durumu güncelle
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Takip durduruldu")
        
    def _process_video(self):
        """Video işleme döngüsü (arka plan thread'i) - OPTIMIZE EDİLMİŞ"""
        frame_count = 0
        start_time = time.time()
        skip_enhancement = False  # Performans için
        last_detections = []  # Frame atlama için
        
        while self.is_running:
            ret, frame = self.video_capture.read()
            
            if not ret:
                if isinstance(self.video_source, str):  # Video dosyası
                    # Video bittiğinde loop yap
                    if self.loop_video.get():
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.is_running = False
                        self.root.after(0, self._stop_tracking)
                continue
                
            frame_count += 1
            
            # Frame atlama - performans için
            if frame_count % self.process_every_n_frames != 0:
                continue
            
            # Görüntü iyileştirme - performans için minimal
            if self.enhance_image.get() and frame_count % 10 == 0:  # Her 10 frame'de bir
                # Sadece basit kontrast ayarı
                frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
                
            # Hızlı tespit - güven eşiğini güncelle
            self.detector.confidence_threshold = self.confidence_var.get()
            detections = self.detector.detect(frame)
            
            # Takip devamlılığı için iyileştirme
            if not detections:
                # Tespit yoksa bile tracker'ı güncelle (kayıp frame olarak)
                if self.tracker:
                    if isinstance(self.tracker, (DeepSortTracker, DroneSpecificTracker)):
                        # Boş detection listesi ile güncelle
                        detections = self.tracker.update([], frame)
                    else:
                        detections = self.tracker.update([])
            else:
                # Yeni tespit varsa sakla
                last_detections = detections
            
            # Takip - sadece tespit varsa
            if self.tracker and detections:
                if isinstance(self.tracker, (DeepSortTracker, DroneSpecificTracker)):
                    # DeepSORT ve DroneSpecific frame parametresi alır
                    detections = self.tracker.update(detections, frame)
                else:
                    detections = self.tracker.update(detections)
                    
            # FPS hesapla - daha sık güncelle
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
            # İstatistikleri güncelle
            self.detection_count = len(detections)
            self.track_count = len([d for d in detections if 'track_id' in d])
            
            # Görüntü çizimi - optimize edilmiş
            display_frame = frame
            
            # Basit çizim (performans için)
            if self.show_detections.get() and detections:
                display_frame = self._fast_draw_detections(display_frame, detections)
                
            if self.show_tracks.get() and self.tracker:
                display_frame = self._draw_tracks(display_frame, detections)
                
            # Hareket maskesini göster
            if self.show_motion_mask.get() and hasattr(self.detector, 'bg_subtractor'):
                display_frame = self._draw_motion_mask(display_frame)
                
            # Kuyruğa ekle - drop frame if full
            try:
                # Eski frame'leri temizle
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
                self.frame_queue.put((display_frame, detections), block=False)
            except queue.Full:
                pass
                
    def _fast_draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Hızlı tespit çizimi - gelişmiş bilgilerle"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Sınıfa göre renk ve etiket
            class_name = det.get('class', 'unknown')
            aerial_class = det.get('aerial_class', class_name)
            
            if aerial_class == 'bird' or class_name == 'bird':
                color = (0, 165, 255)  # Turuncu - kuş
                display_name = "BIRD"
            elif aerial_class == 'airplane':
                color = (255, 0, 255)  # Mor - uçak
                display_name = "AIRPLANE"
            elif aerial_class == 'helicopter':
                color = (255, 128, 0)  # Mavi-yeşil - helikopter
                display_name = "HELICOPTER"
            elif aerial_class == 'drone' or class_name == 'drone':
                # Mesafeye göre renk (drone için)
                range_cat = det.get('range', 'medium')
                if range_cat == 'near':
                    color = (0, 0, 255)  # Kırmızı - yakın
                elif range_cat == 'medium':
                    color = (0, 255, 255)  # Sarı - orta
                else:
                    color = (0, 255, 0)  # Yeşil - uzak
                display_name = "DRONE"
            else:
                color = (128, 128, 128)  # Gri - bilinmeyen
                display_name = "UNKNOWN"
            
            # Dikdörtgen - daha kalın
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Detaylı etiket
            label_lines = []
            
            # Track ID ve nesne tipi
            if 'track_id' in det:
                label_lines.append(f"ID: {det['track_id']}")
            
            # Nesne tipi etiketi
            label_lines.append(display_name)
            
            # Güven skoru (drone için)
            if aerial_class == 'drone' or class_name == 'drone':
                label_lines.append(f"Conf: {conf:.2f}")
            
            # Mesafe bilgisi
            if 'distance' in det:
                label_lines.append(f"Dist: {det['distance']:.0f}m")
                
            # Takip güvenilirliği
            if 'tracking_confidence' in det:
                label_lines.append(f"Track: {det['tracking_confidence']:.2f}")
                
            # Hareket durumu
            if det.get('is_hovering', False):
                label_lines.append("HOVERING")
            elif 'motion_pattern' in det:
                label_lines.append(det['motion_pattern'].upper())
            
            # Etiketleri yaz - daha büyük font
            y_offset = y1 - 5
            for i, line in enumerate(label_lines):
                # ID satırı için daha büyük font
                if "ID:" in line:
                    cv2.putText(frame, line, (x1, y_offset - i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(frame, line, (x1, y_offset - i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                           
            # Merkez işareti
            if 'center' in det:
                cx, cy = det['center']
                cv2.circle(frame, (cx, cy), 3, color, -1)
                
            # Yörünge tahmini
            if 'trajectory' in det and len(det['trajectory']) > 1:
                points = det['trajectory']
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 255, 0), 1)
                    
        return frame
                
    def _update_display(self):
        """Görüntüleme güncelleme döngüsü (UI thread'i) - OPTIMIZE EDİLMİŞ"""
        try:
            # Kuyruktan frame al
            frame, detections = self.frame_queue.get(block=False)
            
            # Canvas boyutuna göre hızlı yeniden boyutlandır
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # macOS için daha iyi interpolation
                interpolation = cv2.INTER_LINEAR if self.is_mac else cv2.INTER_NEAREST
                
                h, w = frame.shape[:2]
                scale = min(canvas_width/w, canvas_height/h) * 0.9  # %90 ölçek
                new_width = int(w * scale)
                new_height = int(h * scale)
                
                # Resize - macOS için daha iyi kalite
                frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
                
                # BGR'den RGB'ye çevir
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # PIL Image'e çevir
                image = Image.fromarray(frame_rgb)
                
                # macOS Retina display desteği
                if self.is_mac:
                    try:
                        # Retina için yüksek çözünürlük
                        photo = ImageTk.PhotoImage(image)
                    except:
                        photo = ImageTk.PhotoImage(image)
                else:
                    photo = ImageTk.PhotoImage(image)
                
                # Canvas'ı güncelle
                self.video_canvas.delete("all")
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.video_canvas.image = photo
                
            # İstatistikleri güncelle
            self._update_stats(detections)
            
        except queue.Empty:
            pass
            
        # Devam et - macOS için optimize edilmiş güncelleme
        if self.is_running:
            # macOS için daha düşük refresh rate (performance)
            refresh_rate = 50 if self.is_mac else 33  # 20 FPS vs 30 FPS
            self.root.after(refresh_rate, self._update_display)
            
    def _draw_tracks(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Takip çizgilerini çiz"""
        # Takip geçmişlerini sakla
        if not hasattr(self, 'track_histories'):
            self.track_histories = {}
            
        # Her tespit için
        for det in detections:
            if 'track_id' not in det:
                continue
                
            track_id = det['track_id']
            center = det['center']
            
            # Geçmişe ekle
            if track_id not in self.track_histories:
                self.track_histories[track_id] = []
                
            self.track_histories[track_id].append(center)
            
            # Maksimum 30 nokta tut
            if len(self.track_histories[track_id]) > 30:
                self.track_histories[track_id].pop(0)
                
            # Çizgiyi çiz
            points = self.track_histories[track_id]
            if len(points) > 1:
                for i in range(1, len(points)):
                    # Renk gradyanı (eski->yeni: mavi->yeşil)
                    color_factor = i / len(points)
                    color = (
                        int(255 * (1 - color_factor)),  # B
                        int(255 * color_factor),         # G
                        0                                # R
                    )
                    
                    # Kalınlık gradyanı
                    thickness = int(1 + 3 * color_factor)
                    
                    # Çizgi çiz
                    cv2.line(frame, points[i-1], points[i], color, thickness)
                    
        # Eski takipleri temizle
        active_ids = [d['track_id'] for d in detections if 'track_id' in d]
        self.track_histories = {k: v for k, v in self.track_histories.items() 
                                if k in active_ids}
        
        return frame
        
    def _draw_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Hareket maskesini göster"""
        if not hasattr(self.detector, 'bg_subtractor'):
            return frame
            
        if len(self.detector.bg_subtractor.motion_masks) == 0:
            return frame
            
        # Son hareket maskesini al
        motion_mask = self.detector.bg_subtractor.motion_masks[-1]
        
        # Boyut kontrolü
        if motion_mask.shape[:2] != frame.shape[:2]:
            motion_mask = cv2.resize(motion_mask, (frame.shape[1], frame.shape[0]))
        
        # Maskeyi renklendir (yeşil)
        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 1] = motion_mask  # Yeşil kanal
        
        # Orijinal frame ile karıştır
        result = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        
        # Hareket olan bölgelerin kenarlarını çiz
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)  # Sarı konturlar
        
        return result
        
    def _update_stats(self, detections: List[Dict]):
        """İstatistikleri güncelle"""
        # FPS
        self.fps_label.config(text=f"{self.fps:.1f}")
        
        # Tespit sayısı
        self.detection_label.config(text=str(self.detection_count))
        
        # Takip sayısı
        self.track_label.config(text=str(self.track_count))
        
        # Tespit listesini güncelle
        self.detection_tree.delete(*self.detection_tree.get_children())
        
        for det in detections:
            if 'track_id' in det:
                # Mesafe bilgisi varsa ekle
                if 'distance' in det and 'range' in det:
                    # EDGS-YOLOv8 + DroneSpecific bilgileri
                    range_info = f"{det['range']} ({det['distance']:.0f}m)"
                    track_conf = f"{det.get('tracking_confidence', 1.0):.2f}"
                    
                    self.detection_tree.insert(
                        "",
                        "end",
                        values=(
                            det['track_id'],
                            f"Drone [{range_info}]",
                            f"{det['confidence']:.2f} / {track_conf}",
                            det.get('age', 0)
                        ),
                        tags=('near',) if det['range'] == 'near' else 
                             ('far',) if det['range'] == 'far' else 
                             ('hovering',) if det.get('is_hovering', False) else ()
                    )
                # Quick-UAV OC-SORT için ek bilgiler
                elif 'motion_pattern' in det:
                    pattern_info = f"{det['motion_pattern']} ({det.get('pattern_confidence', 0):.1f})"
                    speed_info = f"{det.get('speed_estimate', 0):.1f} px/f"
                    alt_info = f"{det.get('altitude_estimate', 0):.1f}m"
                    
                    # Nesne tipini belirle
                    aerial_class = det.get('aerial_class', det.get('class', 'unknown'))
                    display_name = {
                        'drone': 'DRONE',
                        'airplane': 'AIRPLANE',
                        'helicopter': 'HELICOPTER',
                        'bird': 'BIRD',
                        'unknown': 'UNKNOWN'
                    }.get(aerial_class.lower(), aerial_class.upper())
                    
                    self.detection_tree.insert(
                        "",
                        "end",
                        values=(
                            det['track_id'],
                            display_name,
                            f"{det['confidence']:.2f}",
                            det.get('age', 0)
                        ),
                        tags=('hovering',) if det.get('is_hovering', False) else ()
                    )
                else:
                    # Diğer takip algoritmaları için standart
                    # Nesne tipini belirle
                    aerial_class = det.get('aerial_class', det.get('class', 'unknown'))
                    display_name = {
                        'drone': 'DRONE',
                        'airplane': 'AIRPLANE',
                        'helicopter': 'HELICOPTER',
                        'bird': 'BIRD',
                        'unknown': 'UNKNOWN'
                    }.get(aerial_class.lower(), aerial_class.upper())
                    
                    self.detection_tree.insert(
                        "",
                        "end",
                        values=(
                            det['track_id'],
                            display_name,
                            f"{det['confidence']:.2f}",
                            det.get('age', 0)
                        )
                    )
                    
        # Drone'ları mesafeye göre vurgula
        self.detection_tree.tag_configure('near', background='#ffcccc')  # Kırmızımsı - yakın
        self.detection_tree.tag_configure('far', background='#ccffcc')   # Yeşilimsi - uzak
        self.detection_tree.tag_configure('hovering', background='#ffeeee')  # Açık sarı - hovering
                
    def _update_confidence(self, value):
        """Güven eşiği güncellemesi"""
        conf_value = float(value)
        self.confidence_label.config(text=f"{conf_value:.2f}")
        
    def _update_detector_settings(self):
        """Detector ayarlarını güncelle"""
        if hasattr(self, 'detector'):
            self.detector.enable_edgs = self.enable_edgs.get()
            self.detector.multi_scale = self.enable_multiscale.get()
            
            # Multi-scale için scales güncelle
            if self.detector.multi_scale:
                self.detector.scales = [0.75, 1.0]
            else:
                self.detector.scales = [1.0]
                
            logger.info(f"Detector settings updated - EDGS: {self.detector.enable_edgs}, Multi-scale: {self.detector.multi_scale}")
        
    def _save_results(self):
        """Sonuçları kaydet"""
        if not self.is_running:
            messagebox.showwarning("Uyarı", "Önce takibi başlatın!")
            return
            
        # Kayıt dosyası seç
        filename = filedialog.asksaveasfilename(
            title="Sonuçları Kaydet",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )
        
        if filename:
            # TODO: Video kaydetme implementasyonu
            messagebox.showinfo("Bilgi", "Video kaydı özelliği yakında eklenecek!")
            
    def run(self):
        """Uygulamayı çalıştır"""
        # Pencere kapatma eventi
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # macOS için ek optimizasyonlar
        if self.is_mac:
            # macOS için app focus handling
            self.root.lift()
            self.root.attributes('-topmost', False)
            
            # macOS için window state
            try:
                self.root.state('normal')
            except:
                pass
        
        # Ana döngü
        self.root.mainloop()
        
    def _on_closing(self):
        """Pencere kapatılırken"""
        # Takibi durdur
        if self.is_running:
            self._stop_tracking()
            
        # Pencereyi kapat
        self.root.destroy()