import os
import gc
import time
import uuid
import logging
from datetime import datetime
import torch
from config import Config
from database import get_db
from services.preprocessor import Preprocessor
from services.debris_detector import DebrisDetector
from services.anomaly_detector import AnomalyDetector
from services.result_visualizer import ResultVisualizer

logger = logging.getLogger(__name__)

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

            # 3. Model Inference (real detection and anomaly scoring)
            # Each model call is individually protected to prevent total pipeline failure
            import cv2

            debris_res = {'detections': []}
            detections = []

            if self.debris_detector.is_available:
                try:
                    # Primary pass on genuine raw sonar imagery (preserves natural acoustic speckle)
                    debris_res = self.debris_detector.detector.predict(
                        input_path,
                        conf=self.debris_detector.conf_threshold,
                        save_annotated=True,
                        output_dir=proc_folder
                    )
                    detections = debris_res.get('detections', [])

                    # Secondary pass: if raw image had no detections, evaluate on preprocessed file
                    # Use file path (not numpy array) to avoid grayscale/BGR channel mismatch
                    if len(detections) == 0:
                        proc_debris_res = self.debris_detector.detector.predict(
                            proc_abs_path,
                            conf=self.debris_detector.conf_threshold,
                            save_annotated=True,
                            output_dir=proc_folder
                        )
                        if proc_debris_res.get('detections'):
                            debris_res = proc_debris_res
                            detections = proc_debris_res.get('detections', [])
                except Exception as det_err:
                    logger.warning(f"Debris detection failed (non-fatal): {det_err}")

            # Free memory between model calls to prevent OOM on constrained systems
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            anomaly_res = {'anomalies': []}
            anomalies = []

            if self.anomaly_detector.is_available:
                try:
                    anomaly_res = self.anomaly_detector.detect_full(
                        proc_img,
                        save_visuals=True,
                        output_dir=proc_folder
                    )
                    anomalies = anomaly_res.get('anomalies', [])
                except Exception as anom_err:
                    logger.warning(f"Anomaly detection failed (non-fatal): {anom_err}")

            # Free memory after all inference
            gc.collect()

            # 4. Visualization & Annotations (RED marked bounding boxes & structural shapes)
            annotated_abs_path = debris_res.get('annotated_image_path')
            if not annotated_abs_path or not os.path.exists(str(annotated_abs_path)):
                annotated_img = self.visualizer.draw_detections(proc_img, detections)
                if anomalies:
                    annotated_img = self.visualizer.highlight_anomalies(annotated_img, anomalies)

                annotated_filename = f"annotated_{unique_token}.png"
                annotated_abs_path = os.path.join(proc_folder, annotated_filename)
                cv2.imwrite(annotated_abs_path, annotated_img)
            elif anomalies:
                ann_img = cv2.imread(annotated_abs_path)
                if ann_img is not None:
                    ann_img = self.visualizer.highlight_anomalies(ann_img, anomalies)
                    cv2.imwrite(annotated_abs_path, ann_img)

            annotated_rel_path = os.path.relpath(annotated_abs_path, Config.BASE_DIR).replace('\\', '/')

            # Anomaly Heatmap Overlay path
            overlay_rel_path = None
            if anomaly_res.get('overlay_image_path') and os.path.exists(anomaly_res['overlay_image_path']):
                overlay_rel_path = os.path.relpath(anomaly_res['overlay_image_path'], Config.BASE_DIR).replace('\\', '/')

            # Generate side-by-side comparison (Original vs RED-marked AI Detection)
            comp_filename = f"compare_{unique_token}.png"
            comp_abs_path = os.path.join(proc_folder, comp_filename)
            try:
                self.visualizer.build_comparison(input_path, annotated_abs_path, comp_abs_path, right_title="AI NEURAL DETECTION (RED MARKED)")
            except Exception as comp_err:
                logger.warning(f"Comparison image generation failed (non-fatal): {comp_err}")

            # 5. Save Detections and Anomalies
            if detections:
                self.db.save_detections(run_id, detections)
            if anomalies:
                self.db.save_anomalies(run_id, anomalies)

            # 6. Complete Run Record with parameters including overlay path
            elapsed_sec = round(time.time() - start_time, 3)
            run_params = dict(params_used) if params_used else {}
            if overlay_rel_path:
                run_params['anomaly_overlay_path'] = overlay_rel_path
            if anomaly_res.get('anomaly_score') is not None:
                run_params['anomaly_score'] = float(anomaly_res['anomaly_score'])
            if anomaly_res.get('threshold_used') is not None:
                run_params['threshold_used'] = float(anomaly_res['threshold_used'])

            self.db.complete_analysis_run(
                run_id=run_id,
                processed_path=proc_rel_path,
                annotated_path=annotated_rel_path,
                debris_count=len(detections),
                anomaly_count=len(anomalies),
                processing_time_sec=elapsed_sec,
                status='COMPLETED',
                preprocess_params=run_params
            )
            self.db.update_image_status(image_id, 'ANALYZED')

            return self.db.get_run(run_id)

        except Exception as e:
            elapsed_sec = round(time.time() - start_time, 3)
            error_msg = str(e)
            logger.error(f"Analysis pipeline failed for image {image_id}: {error_msg}")
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
