import os
import time
import uuid
from datetime import datetime
from config import Config
from database import get_db
from services.preprocessor import Preprocessor
from services.debris_detector import DebrisDetector
from services.anomaly_detector import AnomalyDetector
from services.result_visualizer import ResultVisualizer

class AnalysisPipeline:
    """
    Coordinates the complete MarineVision analytical workflow:
    1. Preprocessing (OpenCV CLAHE, bilateral filtering, dynamic range normalization)
    2. Model availability check & execution (Debris Detection & Anomaly Detection)
    3. Visualization generation (Annotations & side-by-side comparison)
    4. Database record persistence with timing benchmarks
    """
    def __init__(self, db=None):
        self.db = db or get_db()
        self.preprocessor = Preprocessor(
            denoise_method=Config.PREPROCESS_DEFAULTS['denoise_method'],
            contrast_method=Config.PREPROCESS_DEFAULTS['contrast_method'],
            clahe_clip_limit=Config.PREPROCESS_DEFAULTS['clahe_clip_limit'],
            clahe_tile_grid=Config.PREPROCESS_DEFAULTS['clahe_tile_grid'],
            normalize=Config.PREPROCESS_DEFAULTS['normalize']
        )
        self.debris_detector = DebrisDetector()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = ResultVisualizer()

    def run(self, image_id):
        """
        Executes an analysis run for the specified image_id.
        Returns the completed run dictionary.
        """
        image = self.db.get_image(image_id)
        if not image:
            raise ValueError(f"Image ID {image_id} does not exist in the database.")

        start_time = time.time()
        
        # Check active models in DB or filesystem
        debris_model_id = None
        anomaly_model_id = None
        if self.debris_detector.is_available:
            debris_model = self.db.get_active_model('DEBRIS_DETECTOR')
            if debris_model:
                debris_model_id = debris_model['model_id']

        if self.anomaly_detector.is_available:
            anomaly_model = self.db.get_active_model('ANOMALY_DETECTOR')
            if anomaly_model:
                anomaly_model_id = anomaly_model['model_id']

        # 1. Create run record
        run_id = self.db.create_analysis_run(
            image_id=image_id,
            debris_model_id=debris_model_id,
            anomaly_model_id=anomaly_model_id,
            preprocess_params=Config.PREPROCESS_DEFAULTS
        )
        
        # Update image status to PROCESSING
        self.db.update_image_status(image_id, 'PROCESSING')

        try:
            # 2. Preprocess Sonar Image
            try:
                from flask import current_app
                proc_folder = current_app.config.get('PROCESSED_FOLDER', Config.PROCESSED_FOLDER)
            except Exception:
                proc_folder = Config.PROCESSED_FOLDER

            unique_token = f"{image_id}_{uuid.uuid4().hex[:8]}"
            proc_filename = f"proc_{unique_token}.png"
            proc_abs_path = os.path.join(proc_folder, proc_filename)
            
            # Resolve image stored path
            input_path = image['stored_path']
            if not os.path.isabs(input_path):
                input_path = os.path.join(Config.BASE_DIR, input_path)

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Original image file not found at {input_path}")

            out_path, params_used, proc_img = self.preprocessor.process(input_path, proc_abs_path)
            proc_rel_path = os.path.relpath(proc_abs_path, Config.BASE_DIR).replace('\\', '/')

            # 3. Model Inference (real detection if model available)
            detections = self.debris_detector.detect(proc_img)
            anomalies = self.anomaly_detector.detect(proc_img)

            # 4. Visualization & Annotations
            annotated_rel_path = proc_rel_path
            if detections or anomalies:
                annotated_img = proc_img.copy()
                if detections:
                    annotated_img = self.visualizer.draw_detections(annotated_img, detections)
                if anomalies:
                    annotated_img = self.visualizer.highlight_anomalies(annotated_img, anomalies)

                annotated_filename = f"annotated_{unique_token}.png"
                annotated_abs_path = os.path.join(proc_folder, annotated_filename)
                import cv2
                cv2.imwrite(annotated_abs_path, annotated_img)
                annotated_rel_path = os.path.relpath(annotated_abs_path, Config.BASE_DIR).replace('\\', '/')

            # Generate side-by-side comparison
            comp_filename = f"compare_{unique_token}.png"
            comp_abs_path = os.path.join(proc_folder, comp_filename)
            self.visualizer.build_comparison(input_path, proc_abs_path, comp_abs_path)

            # 5. Save Detections and Anomalies
            if detections:
                self.db.save_detections(run_id, detections)
            if anomalies:
                self.db.save_anomalies(run_id, anomalies)

            # 6. Complete Run Record
            elapsed_sec = round(time.time() - start_time, 3)
            self.db.complete_analysis_run(
                run_id=run_id,
                processed_path=proc_rel_path,
                annotated_path=annotated_rel_path,
                debris_count=len(detections),
                anomaly_count=len(anomalies),
                processing_time_sec=elapsed_sec,
                status='COMPLETED'
            )
            self.db.update_image_status(image_id, 'ANALYZED')

            return self.db.get_run(run_id)

        except Exception as e:
            elapsed_sec = round(time.time() - start_time, 3)
            error_msg = str(e)
            self.db.complete_analysis_run(
                run_id=run_id,
                processed_path=None,
                annotated_path=None,
                debris_count=0,
                anomaly_count=0,
                processing_time_sec=elapsed_sec,
                status='FAILED',
                error_message=error_msg
            )
            self.db.update_image_status(image_id, 'FAILED')
            raise e
