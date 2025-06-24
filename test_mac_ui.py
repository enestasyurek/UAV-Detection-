#!/usr/bin/env python3
"""
macOS Tkinter UI Test
Basit test penceresi - macOS'ta UI bileşenlerinin düzgün çalışıp çalışmadığını kontrol eder
"""

import tkinter as tk
from tkinter import ttk
import platform

def test_mac_ui():
    """macOS UI test fonksiyonu"""
    print("macOS UI Test başlatılıyor...")
    
    # Platform kontrolü
    current_platform = platform.system()
    print(f"Platform: {current_platform}")
    
    if current_platform != 'Darwin':
        print("Bu test sadece macOS için tasarlandı!")
        return
    
    # Ana pencere
    root = tk.Tk()
    root.title("macOS UI Test")
    root.geometry("800x600")
    root.configure(bg="#f5f5f5")
    
    print("Ana pencere oluşturuldu")
    
    # Tema test
    style = ttk.Style()
    available_themes = style.theme_names()
    print(f"Mevcut temalar: {available_themes}")
    
    if 'aqua' in available_themes:
        style.theme_use('aqua')
        print("Aqua teması kullanılıyor")
    else:
        style.theme_use('default')
        print("Default tema kullanılıyor")
    
    # Ana frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    print("Ana frame oluşturuldu")
    
    # Test label
    title_label = ttk.Label(main_frame, text="macOS UI Test", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=10)
    print("Başlık etiketi oluşturuldu")
    
    # Test button
    test_button = ttk.Button(main_frame, text="Test Butonu", command=lambda: print("Buton çalışıyor!"))
    test_button.pack(pady=5)
    print("Test butonu oluşturuldu")
    
    # Test frame with controls
    control_frame = ttk.LabelFrame(main_frame, text="Kontroller", padding=10)
    control_frame.pack(fill=tk.X, pady=10)
    print("Kontrol frame'i oluşturuldu")
    
    # Test combobox
    combo_var = tk.StringVar(value="Seçenek 1")
    test_combo = ttk.Combobox(control_frame, textvariable=combo_var, 
                             values=["Seçenek 1", "Seçenek 2", "Seçenek 3"], 
                             state="readonly")
    test_combo.pack(pady=5)
    print("Combobox oluşturuldu")
    
    # Test scale
    scale_var = tk.DoubleVar(value=0.5)
    test_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=scale_var, orient="horizontal")
    test_scale.pack(fill=tk.X, pady=5)
    print("Scale oluşturuldu")
    
    # Canvas test
    canvas_frame = ttk.LabelFrame(main_frame, text="Canvas Test", padding=10)
    canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    test_canvas = tk.Canvas(canvas_frame, width=400, height=200, bg="white")
    test_canvas.pack(pady=5)
    
    # Canvas'a test çizimi
    test_canvas.create_rectangle(50, 50, 150, 100, fill="blue", outline="navy")
    test_canvas.create_text(200, 75, text="Canvas Test", font=("Helvetica", 14))
    print("Canvas oluşturuldu ve test çizimi yapıldı")
    
    # Treeview test
    tree_frame = ttk.Frame(main_frame)
    tree_frame.pack(fill=tk.X, pady=10)
    
    test_tree = ttk.Treeview(tree_frame, columns=("Col1", "Col2"), show="headings", height=3)
    test_tree.heading("Col1", text="Sütun 1")
    test_tree.heading("Col2", text="Sütun 2")
    test_tree.pack()
    
    # Test data
    test_tree.insert("", "end", values=("Test 1", "Değer 1"))
    test_tree.insert("", "end", values=("Test 2", "Değer 2"))
    print("Treeview oluşturuldu")
    
    # Status bar
    status_frame = ttk.Frame(root)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    status_label = ttk.Label(status_frame, text="macOS UI Test - Hazır", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
    print("Status bar oluşturuldu")
    
    # Force update
    root.update_idletasks()
    print(f"Pencere boyutu: {root.winfo_width()}x{root.winfo_height()}")
    print(f"Pencere konumu: {root.winfo_x()}, {root.winfo_y()}")
    
    print("UI test tamamlandı - pencere açılıyor...")
    
    # Pencereyi göster
    root.lift()
    root.focus_force()
    
    # Test tamamlandı mesajı
    def show_success():
        success_label = ttk.Label(main_frame, text="✅ UI Test Başarılı!", font=("Helvetica", 14, "bold"), foreground="green")
        success_label.pack(pady=10)
        
    root.after(1000, show_success)
    
    # Ana loop
    root.mainloop()

if __name__ == "__main__":
    test_mac_ui()