import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    BASE_DIR = str(BASE_DIR)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'marinevision-sonar-intel-secret-key-2026')
    
    # Storage directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
    REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
    INSTANCE_FOLDER = os.path.join(BASE_DIR, 'instance')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Database
    DATABASE_PATH = os.path.join(INSTANCE_FOLDER, 'marinevision.db')
    SCHEMA_PATH = os.path.join(BASE_DIR, 'schema.sql')
    
    # Upload limits and constraints
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}
    MIN_IMAGE_WIDTH = 64
    MIN_IMAGE_HEIGHT = 64
    
    # Model configuration paths
    YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'yolo_debris.pt')
    ANOMALY_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'autoencoder_anomaly.pth')
    
    # Preprocessing default parameters
    PREPROCESS_DEFAULTS = {
        'denoise_method': 'bilateral',
        'contrast_method': 'clahe',
        'clahe_clip_limit': 2.5,
        'clahe_tile_grid': (8, 8),
        'clahe_tile_grid_size': (8, 8),
        'normalize': True,
        'standardize_size': False,
        'target_width': 1024,
        'target_height': 1024
    }
    
    # Project and university metadata
    PROJECT_NAME = "MarineVision"
    PROJECT_TAGLINE = "Underwater Intelligence & Sonar Analytics"
    PROJECT_TITLE = "AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery"
    INSTITUTION = "Institute of Advanced Research (IAR University)"
    DEPARTMENT = "BTech CE Semester 7th (AI)"
    GROUP_NAME = "CodeQuacks"
    GROUP_NO = "55"
    SCHOLARS = [
        {"name": "Rudra Mistry", "reg_no": "14311"},
        {"name": "Jaimin Thakkar", "reg_no": "14159"},
        {"name": "Meet Patel", "reg_no": "14486"}
    ]
    SUPERVISOR = "Ms. Pooja Chawla"
