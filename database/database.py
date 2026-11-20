import sqlite3
import os
import json
from datetime import datetime
from config import Config

class DatabaseManager:
    """
    Manages all SQLite database operations for the MarineVision application.
    Enforces SQLite Foreign Keys (PRAGMA foreign_keys = ON) and parameterized queries.
    """
    def __init__(self, db_path=None, schema_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.schema_path = schema_path or Config.SCHEMA_PATH
        
        # Ensure instance directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self):
        """Creates and returns an active SQLite connection with row_factory and foreign keys."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """Initializes database tables and indexes from schema.sql."""
        with self.get_connection() as conn:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema_script = f.read()
            conn.executescript(schema_script)
            conn.commit()

    # =========================================================================
    # Images
    # =========================================================================

    def save_image(self, original_filename, stored_path, file_format, file_size_kb, width_px, height_px, status='UPLOADED'):
        """Inserts a new image record and returns image_id."""
        # Normalize file_format to match CHECK constraint (JPG, PNG, TIFF)
        fmt = file_format.upper().replace('JPEG', 'JPG').replace('TIF', 'TIFF')
        if fmt not in ('JPG', 'PNG', 'TIFF'):
            fmt = 'JPG'

        query = """
            INSERT INTO images (original_filename, stored_path, file_format, file_size_kb, width_px, height_px, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (original_filename, stored_path, fmt, file_size_kb, width_px, height_px, status))
            conn.commit()
            return cursor.lastrowid

    def get_image(self, image_id):
        """Retrieves an image record by ID."""
        query = "SELECT * FROM images WHERE image_id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_image_status(self, image_id, status):
        """Updates the processing status of an image."""
        query = "UPDATE images SET status = ? WHERE image_id = ?"
        with self.get_connection() as conn:
            conn.execute(query, (status, image_id))
            conn.commit()

    # =========================================================================
    # ML Models
    # =========================================================================

    def register_model(self, model_name, model_type, version, weights_path, threshold=None, trained_on=None, is_active=1):
        """Registers an ML model record if weights actually exist."""
        query = """
            INSERT INTO ml_models (model_name, model_type, version, weights_path, threshold, trained_on, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (model_name, model_type, version, weights_path, threshold, trained_on, is_active))
            conn.commit()
            return cursor.lastrowid

    def get_active_model(self, model_type):
        """Gets active model of given type (DEBRIS_DETECTOR, ANOMALY_DETECTOR)."""
        query = "SELECT * FROM ml_models WHERE model_type = ? AND is_active = 1 ORDER BY model_id DESC LIMIT 1"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (model_type,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_models(self):
        """Returns all registered models."""
        query = "SELECT * FROM ml_models ORDER BY model_id ASC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def ensure_model_registered(self, model_name, model_type, version, weights_path, threshold=None, trained_on=None, is_active=1):
        """Ensures an active model record exists for the given type and weights path."""
        existing = self.get_active_model(model_type)
        if existing and existing['weights_path'] == weights_path:
            query = "UPDATE ml_models SET threshold = ?, version = ?, trained_on = ? WHERE model_id = ?"
            with self.get_connection() as conn:
                conn.execute(query, (threshold, version, trained_on, existing['model_id']))
                conn.commit()
            return existing['model_id']
        else:
            return self.register_model(model_name, model_type, version, weights_path, threshold, trained_on, is_active)

    # =========================================================================
    # Analysis Runs
    # =========================================================================

    def create_analysis_run(self, image_id, debris_model_id=None, anomaly_model_id=None, preprocess_params=None):
        """Creates a new analysis run entry in RUNNING state."""
        params_json = json.dumps(preprocess_params) if preprocess_params else None
        query = """
            INSERT INTO analysis_runs (
                image_id, debris_model_id, anomaly_model_id, preprocess_params, status, started_at
            ) VALUES (?, ?, ?, ?, 'RUNNING', CURRENT_TIMESTAMP)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (image_id, debris_model_id, anomaly_model_id, params_json))
            conn.commit()
            return cursor.lastrowid

    def complete_analysis_run(self, run_id, processed_path, annotated_path, debris_count, anomaly_count, processing_time_sec, status='COMPLETED', error_message=None, preprocess_params=None):
        """Updates analysis run upon completion or failure."""
        params_json = json.dumps(preprocess_params) if preprocess_params else None
        if params_json:
            query = """
                UPDATE analysis_runs
                SET processed_path = ?,
                    annotated_path = ?,
                    debris_count = ?,
                    anomaly_count = ?,
                    processing_time_sec = ?,
                    status = ?,
                    error_message = ?,
                    preprocess_params = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """
            with self.get_connection() as conn:
                conn.execute(query, (
                    processed_path, annotated_path, debris_count, anomaly_count,
                    processing_time_sec, status, error_message, params_json, run_id
                ))
                conn.commit()
        else:
            query = """
                UPDATE analysis_runs
                SET processed_path = ?,
                    annotated_path = ?,
                    debris_count = ?,
                    anomaly_count = ?,
                    processing_time_sec = ?,
                    status = ?,
                    error_message = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """
            with self.get_connection() as conn:
                conn.execute(query, (
                    processed_path, annotated_path, debris_count, anomaly_count,
                    processing_time_sec, status, error_message, run_id
                ))
                conn.commit()

    def get_run(self, run_id):
        """Gets analysis run by run_id with image metadata joined."""
        query = """
            SELECT r.*, i.original_filename, i.stored_path, i.file_format, i.file_size_kb,
                   i.width_px, i.height_px, i.uploaded_at as image_uploaded_at,
                   dm.model_name as debris_model_name, dm.version as debris_model_version,
                   am.model_name as anomaly_model_name, am.version as anomaly_model_version
            FROM analysis_runs r
            JOIN images i ON r.image_id = i.image_id
            LEFT JOIN ml_models dm ON r.debris_model_id = dm.model_id
            LEFT JOIN ml_models am ON r.anomaly_model_id = am.model_id
            WHERE r.run_id = ?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_run_for_image(self, image_id):
        """Gets the most recent analysis run for an image."""
        query = """
            SELECT r.*, i.original_filename, i.stored_path, i.file_format, i.file_size_kb,
                   i.width_px, i.height_px, i.uploaded_at as image_uploaded_at,
                   dm.model_name as debris_model_name, dm.version as debris_model_version,
                   am.model_name as anomaly_model_name, am.version as anomaly_model_version
            FROM analysis_runs r
            JOIN images i ON r.image_id = i.image_id
            LEFT JOIN ml_models dm ON r.debris_model_id = dm.model_id
            LEFT JOIN ml_models am ON r.anomaly_model_id = am.model_id
            WHERE r.image_id = ?
            ORDER BY r.run_id DESC LIMIT 1
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_runs_for_image(self, image_id):
        """Returns all analysis runs for an image."""
        query = "SELECT * FROM analysis_runs WHERE image_id = ? ORDER BY run_id DESC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (image_id,))
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Detections & Anomalies
    # =========================================================================

    def save_detections(self, run_id, detections_list):
        """
        Saves real debris detections.
        Each item: {'class_label': str, 'confidence': float, 'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int}
        """
        if not detections_list:
            return
        query = """
            INSERT INTO detections (run_id, class_label, confidence, x_min, y_min, x_max, y_max)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for d in detections_list:
                cursor.execute(query, (
                    run_id, d['class_label'], float(d['confidence']),
                    int(d['x_min']), int(d['y_min']), int(d['x_max']), int(d['y_max'])
                ))
            conn.commit()

    def get_detections(self, run_id):
        """Returns all detections for a run."""
        query = "SELECT * FROM detections WHERE run_id = ? ORDER BY confidence DESC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def save_anomalies(self, run_id, anomalies_list):
        """
        Saves real anomaly detections.
        Each item: {'anomaly_score': float, 'x': int, 'y': int, 'width': int, 'height': int, 'severity': 'LOW'|'MEDIUM'|'HIGH'}
        """
        if not anomalies_list:
            return
        query = """
            INSERT INTO anomalies (run_id, anomaly_score, x, y, width, height, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for a in anomalies_list:
                cursor.execute(query, (
                    run_id, float(a['anomaly_score']),
                    int(a['x']), int(a['y']), int(a['width']), int(a['height']),
                    a['severity'].upper()
                ))
            conn.commit()

    def get_anomalies(self, run_id):
        """Returns all anomalies for a run."""
        query = "SELECT * FROM anomalies WHERE run_id = ? ORDER BY anomaly_score DESC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Dashboard Analytics & History Queries
    # =========================================================================

    def get_dashboard_metrics(self):
        """Computes live analytics counts from actual SQLite database records."""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            # Total images
            c.execute("SELECT COUNT(*) FROM images")
            total_images = c.fetchone()[0]
            
            # Completed analyses
            c.execute("SELECT COUNT(*) FROM analysis_runs WHERE status = 'COMPLETED'")
            completed_analyses = c.fetchone()[0]
            
            # Pending analyses (images uploaded or processing without completed run)
            c.execute("""
                SELECT COUNT(*) FROM images
                WHERE status IN ('UPLOADED', 'PROCESSING')
            """)
            pending_analyses = c.fetchone()[0]
            
            # Total detected debris objects
            c.execute("SELECT COUNT(*) FROM detections")
            detected_debris = c.fetchone()[0]

            # Total detected anomalies
            c.execute("SELECT COUNT(*) FROM anomalies")
            total_anomalies = c.fetchone()[0]

            return {
                'total_images': total_images,
                'completed_analyses': completed_analyses,
                'pending_analyses': pending_analyses,
                'detected_debris': detected_debris,
                'total_anomalies': total_anomalies
            }

    def get_recent_uploads(self, limit=5):
        """Gets recent images with their latest run status and metrics."""
        query = """
            SELECT i.image_id, i.original_filename, i.file_format, i.file_size_kb,
                   i.width_px, i.height_px, i.uploaded_at, i.status as image_status,
                   r.run_id, r.status as run_status, r.debris_count, r.anomaly_count,
                   r.processing_time_sec
            FROM images i
            LEFT JOIN (
                SELECT image_id, MAX(run_id) as max_run_id
                FROM analysis_runs
                GROUP BY image_id
            ) latest ON i.image_id = latest.image_id
            LEFT JOIN analysis_runs r ON latest.max_run_id = r.run_id
            ORDER BY i.image_id DESC
            LIMIT ?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_history(self, search=None, status_filter=None, sort_order='DESC', limit=100):
        """
        Returns full analysis history with filtering and search support.
        """
        query = """
            SELECT i.image_id, i.original_filename, i.stored_path, i.file_format,
                   i.file_size_kb, i.width_px, i.height_px, i.uploaded_at,
                   r.run_id, r.status as run_status, r.debris_count, r.anomaly_count,
                   r.processing_time_sec, r.started_at, r.completed_at,
                   r.processed_path, r.annotated_path
            FROM images i
            LEFT JOIN analysis_runs r ON i.image_id = r.image_id
            WHERE 1=1
        """
        params = []
        if search:
            query += " AND (i.original_filename LIKE ? OR CAST(i.image_id AS TEXT) = ?)"
            search_param = f"%{search.strip()}%"
            params.extend([search_param, search.strip()])
            
        if status_filter:
            if status_filter.upper() == 'COMPLETED':
                query += " AND r.status = 'COMPLETED'"
            elif status_filter.upper() == 'PENDING':
                query += " AND (r.status = 'RUNNING' OR r.status IS NULL OR i.status IN ('UPLOADED', 'PROCESSING'))"
            elif status_filter.upper() == 'FAILED':
                query += " AND (r.status = 'FAILED' OR i.status = 'FAILED')"
                
        order = 'DESC' if sort_order.upper() == 'DESC' else 'ASC'
        query += f" ORDER BY COALESCE(r.started_at, i.uploaded_at) {order} LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

_db_instance = None

def get_db():
    """Singleton getter for DatabaseManager."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
