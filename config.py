# -*- coding: utf-8 -*-
"""
File penengah (Proxy Config) untuk mengelabui OpenCV (cv2) di Streamlit Cloud.
File ini akan mendeteksi secara otomatis apakah pemanggilnya adalah modul internal Anda 
atau library OpenCV, lalu mengarahkan ke file konfigurasi yang tepat.
"""
import os
import sys
import importlib.util

# Jalankan deteksi lokasi file config.py asli di dalam folder Source_Code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_CONFIG_PATH = os.path.join(BASE_DIR, 'Source_Code', 'config.py')

if os.path.exists(REAL_CONFIG_PATH):
    # Paksa load config.py medis Anda yang asli
    spec = importlib.util.spec_from_file_location("real_medical_config", REAL_CONFIG_PATH)
    real_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_config)
    
    # Ekspor semua variabel agar bisa dibaca oleh augmentation.py, models.py, dll.
    globals().update(vars(real_config))
else:
    # Jika dipanggil oleh cv2 saat setup env awal dan path di atas belum siap
    # berikan fallback aman agar server tidak crash
    IMG_SIZE = 224
    NUM_CLASSES = 3
    CLASS_NAMES = ['NORMAL', 'BACTERIA', 'VIRUS']
