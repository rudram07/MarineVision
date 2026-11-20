-- MarineVision SQLite Database Schema
-- Matches System Design specifications from Progress Report II

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL UNIQUE,
    file_format TEXT NOT NULL CHECK (file_format IN ('JPG','PNG','TIFF')),
    file_size_kb REAL NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'UPLOADED'
    CHECK (status IN ('UPLOADED','PROCESSING','ANALYZED','FAILED'))
);

CREATE TABLE IF NOT EXISTS ml_models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('DEBRIS_DETECTOR','ANOMALY_DETECTOR')),
    version TEXT NOT NULL,
    weights_path TEXT NOT NULL,
    threshold REAL,
    trained_on TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    debris_model_id INTEGER REFERENCES ml_models(model_id),
    anomaly_model_id INTEGER REFERENCES ml_models(model_id),
    processed_path TEXT,
    annotated_path TEXT,
    preprocess_params TEXT,
    debris_count INTEGER NOT NULL DEFAULT 0,
    anomaly_count INTEGER NOT NULL DEFAULT 0,
    processing_time_sec REAL,
    status TEXT NOT NULL DEFAULT 'RUNNING'
    CHECK (status IN ('RUNNING','COMPLETED','FAILED')),
    error_message TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    class_label TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    x_min INTEGER NOT NULL,
    y_min INTEGER NOT NULL,
    x_max INTEGER NOT NULL,
    y_max INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    anomaly_score REAL NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('LOW','MEDIUM','HIGH'))
);

CREATE INDEX IF NOT EXISTS idx_runs_image ON analysis_runs(image_id);
CREATE INDEX IF NOT EXISTS idx_det_run ON detections(run_id);
CREATE INDEX IF NOT EXISTS idx_anom_run ON anomalies(run_id);
