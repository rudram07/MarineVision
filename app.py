import os
import uuid
import shutil
from datetime import datetime
import werkzeug
if not hasattr(werkzeug, '__version__'):
    try:
        import importlib.metadata
        werkzeug.__version__ = importlib.metadata.version('werkzeug')
    except Exception:
        werkzeug.__version__ = '3.1.3'

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, send_from_directory, abort
)
from werkzeug.utils import secure_filename

from config import Config
from database import get_db, DatabaseManager
from services import (
    ImageValidator, Preprocessor, DebrisDetector,
    AnomalyDetector, ResultVisualizer, AnalysisPipeline, ReportGenerator
)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure all required runtime directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INSTANCE_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)

    # Initialize Database
    db = DatabaseManager(app.config['DATABASE_PATH'], app.config['SCHEMA_PATH'])
    db.init_db()

    # Initialize Services
    validator = ImageValidator(
        allowed_formats=app.config['ALLOWED_EXTENSIONS'],
        max_size_mb=10,
        min_width=app.config['MIN_IMAGE_WIDTH'],
        min_height=app.config['MIN_IMAGE_HEIGHT']
    )
    pipeline = AnalysisPipeline(db=db)
    report_gen = ReportGenerator(db=db)

    # Register active models in database if checkpoints exist on disk
    try:
        debris_info = pipeline.debris_detector.get_info()
        if debris_info.get('is_available') and debris_info.get('weights_path'):
            db.ensure_model_registered(
                model_name=debris_info.get('model_name', 'YOLO Marine Debris Detector'),
                model_type='DEBRIS_DETECTOR',
                version=debris_info.get('version', 'v1.0'),
                weights_path=debris_info['weights_path'],
                threshold=debris_info.get('threshold', 0.25),
                trained_on='Side-Scan Sonar Debris Dataset (from scratch)'
            )
        anomaly_info = pipeline.anomaly_detector.get_info()
        if anomaly_info.get('is_available') and anomaly_info.get('weights_path'):
            db.ensure_model_registered(
                model_name=anomaly_info.get('model_name', 'CNN Autoencoder Anomaly Detector'),
                model_type='ANOMALY_DETECTOR',
                version=anomaly_info.get('version', 'v1.0'),
                weights_path=anomaly_info['weights_path'],
                threshold=anomaly_info.get('threshold', 0.247),
                trained_on='Verified Normal Seabed Sonar Imagery (from scratch)'
            )
    except Exception as e:
        app.logger.warning(f"Could not auto-register active ML models: {e}")

    # Context processor to pass common branding and model status to all templates
    @app.context_processor
    def inject_global_context():
        debris_info = pipeline.debris_detector.get_info()
        anomaly_info = pipeline.anomaly_detector.get_info()
        return {
            'project_name': Config.PROJECT_NAME,
            'project_tagline': Config.PROJECT_TAGLINE,
            'project_title': Config.PROJECT_TITLE,
            'institution': Config.INSTITUTION,
            'group_name': Config.GROUP_NAME,
            'group_no': Config.GROUP_NO,
            'scholars': Config.SCHOLARS,
            'supervisor': Config.SUPERVISOR,
            'debris_model_available': debris_info['is_available'],
            'debris_model_info': debris_info,
            'anomaly_model_available': anomaly_info['is_available'],
            'anomaly_model_info': anomaly_info,
            'current_year': datetime.now().year
        }

    # =========================================================================
    # Routes
    # =========================================================================

    @app.route('/')
    @app.route('/intro')
    def intro():
        """Page 1: Welcome and Introduction to MarineVision."""
        return render_template('intro.html', active_page='intro')

    @app.route('/dashboard')
    def dashboard():
        """Page 2: Main Dashboard with live SQLite metrics and activity."""
        metrics = db.get_dashboard_metrics()
        recent_uploads = db.get_recent_uploads(limit=6)
        models = db.get_all_models()
        return render_template(
            'dashboard.html',
            active_page='dashboard',
            metrics=metrics,
            recent_uploads=recent_uploads,
            models=models
        )

    @app.route('/analyze')
    def analyze():
        """Page 3: Combined Upload, Processing & Results (Upload state)."""
        return render_template(
            'analyze.html',
            active_page='analyze',
            selected_image=None,
            latest_run=None,
            detections=[],
            anomalies=[]
        )

    @app.route('/analyze/<int:image_id>')
    def analyze_view(image_id):
        """Page 3: Combined Upload, Processing & Results (Viewing specific analysis)."""
        image = db.get_image(image_id)
        if not image:
            flash(f"Image with ID #{image_id} was not found.", "warning")
            return redirect(url_for('analyze'))

        latest_run = db.get_latest_run_for_image(image_id)
        detections = []
        anomalies = []
        anomaly_overlay_path = None
        anomaly_score = None
        threshold_used = None

        if latest_run:
            detections = db.get_detections(latest_run['run_id'])
            anomalies = db.get_anomalies(latest_run['run_id'])

            # Enrich detections with physical shape and structural characteristics
            from services.result_visualizer import OBJECT_STRUCTURAL_PROFILES
            for d in detections:
                raw_cls = d.get('class_label', '')
                prof = OBJECT_STRUCTURAL_PROFILES.get(raw_cls, {})
                d['display_name'] = prof.get('display_name', raw_cls.replace('_', ' ').title())
                d['structural_shape'] = prof.get('shape', 'Acoustic Target Structure')
                d['structural_description'] = prof.get('structure', 'Acoustic contact with distinctive sonar backscatter.')

            if latest_run.get('preprocess_params'):
                try:
                    import json
                    p = json.loads(latest_run['preprocess_params']) if isinstance(latest_run['preprocess_params'], str) else latest_run['preprocess_params']
                    anomaly_overlay_path = p.get('anomaly_overlay_path')
                    anomaly_score = p.get('anomaly_score')
                    threshold_used = p.get('threshold_used')
                except Exception:
                    pass

        return render_template(
            'analyze.html',
            active_page='analyze',
            selected_image=image,
            latest_run=latest_run,
            detections=detections,
            anomalies=anomalies,
            anomaly_overlay_path=anomaly_overlay_path,
            anomaly_score=anomaly_score,
            threshold_used=threshold_used
        )

    @app.route('/upload', methods=['POST'])
    @app.route('/analyze', methods=['POST'])
    def handle_upload():
        """
        Processes image upload:
        - Validates extension, file size, and OpenCV image decoding
        - Saves with secure unique filename to uploads/
        - Records metadata in SQLite
        - Executes complete AnalysisPipeline (preprocessing, AI check, visualization)
        - Redirects or returns JSON pointing to /analyze/<image_id>
        """
        is_ajax = (
            request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
            request.content_type and 'application/json' in request.content_type or
            request.accept_mimetypes.best == 'application/json'
        )

        # 1. Check for file
        if 'image' not in request.files:
            msg = "No image file provided in upload request."
            if is_ajax:
                return jsonify({'success': False, 'error': msg}), 400
            flash(msg, 'danger')
            return redirect(url_for('analyze'))

        file = request.files['image']
        
        # 2. Server-side validation
        val_result = validator.validate_file_object(file)
        if not val_result.is_valid:
            if is_ajax:
                return jsonify({'success': False, 'error': val_result.error_message}), 400
            flash(val_result.error_message, 'danger')
            return redirect(url_for('analyze'))

        # 3. Secure unique filename and storage
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'png'
        unique_token = uuid.uuid4().hex[:12]
        safe_name = f"sonar_{unique_token}.{ext}"
        stored_abs_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        stored_rel_path = os.path.relpath(stored_abs_path, Config.BASE_DIR).replace('\\', '/')

        try:
            file.save(stored_abs_path)
        except Exception as e:
            msg = f"Failed to save uploaded file on server: {str(e)}"
            if is_ajax:
                return jsonify({'success': False, 'error': msg}), 500
            flash(msg, 'danger')
            return redirect(url_for('analyze'))

        # 4. Save metadata in SQLite
        try:
            image_id = db.save_image(
                original_filename=val_result.original_filename,
                stored_path=stored_rel_path,
                file_format=val_result.file_format,
                file_size_kb=val_result.file_size_kb,
                width_px=val_result.width_px,
                height_px=val_result.height_px,
                status='UPLOADED'
            )
        except Exception as e:
            msg = f"Database insertion error: {str(e)}"
            if is_ajax:
                return jsonify({'success': False, 'error': msg}), 500
            flash(msg, 'danger')
            return redirect(url_for('analyze'))

        # 5. Execute Analysis Pipeline
        try:
            run_result = pipeline.run(image_id)
        except Exception as e:
            # Run failed, but image record exists
            flash(f"Image uploaded but analysis pipeline encountered an error: {str(e)}", 'warning')
            if is_ajax:
                return jsonify({
                    'success': True,
                    'image_id': image_id,
                    'warning': str(e),
                    'redirect_url': url_for('analyze_view', image_id=image_id)
                })
            return redirect(url_for('analyze_view', image_id=image_id))

        if is_ajax:
            return jsonify({
                'success': True,
                'image_id': image_id,
                'run_id': run_result['run_id'],
                'redirect_url': url_for('analyze_view', image_id=image_id)
            })

        flash("Sonar image uploaded and preprocessed successfully!", "success")
        return redirect(url_for('analyze_view', image_id=image_id))

    @app.route('/api/load-sample/<sample_name>', methods=['POST'])
    def load_sample(sample_name):
        """Helper to quickly load a realistic synthetic Side-Scan Sonar test image."""
        sample_map = {
            'debris': 'sample_sonar_debris.png',
            'anomaly': 'sample_sonar_anomaly.png'
        }
        filename = sample_map.get(sample_name)
        if not filename:
            return jsonify({'success': False, 'error': 'Invalid sample selected.'}), 400

        source_path = os.path.join(app.root_path, 'static', 'images', 'samples', filename)
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': 'Sample file not found on disk.'}), 404

        # Copy to uploads as new image
        unique_token = uuid.uuid4().hex[:12]
        new_name = f"sonar_sample_{sample_name}_{unique_token}.png"
        dest_abs = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
        dest_rel = os.path.relpath(dest_abs, Config.BASE_DIR).replace('\\', '/')

        shutil.copyfile(source_path, dest_abs)

        # Get file info
        import cv2
        img = cv2.imread(dest_abs)
        h, w = img.shape[:2]
        size_kb = os.path.getsize(dest_abs) / 1024.0

        image_id = db.save_image(
            original_filename=f"sonar_{sample_name}_sample.png",
            stored_path=dest_rel,
            file_format='PNG',
            file_size_kb=round(size_kb, 2),
            width_px=w,
            height_px=h,
            status='UPLOADED'
        )

        # Run pipeline
        pipeline.run(image_id)

        return jsonify({
            'success': True,
            'image_id': image_id,
            'redirect_url': url_for('analyze_view', image_id=image_id)
        })

    @app.route('/history')
    def history():
        """Page 4: Analysis History with search, filter, and export actions."""
        q = request.args.get('q', '').strip()
        status_filter = request.args.get('status', '').strip()
        sort_order = request.args.get('sort', 'DESC').strip()

        history_records = db.get_history(search=q, status_filter=status_filter, sort_order=sort_order)
        return render_template(
            'history.html',
            active_page='history',
            records=history_records,
            q=q,
            status_filter=status_filter,
            sort_order=sort_order
        )

    @app.route('/about')
    def about():
        """Page 5: About the System, Problem Statement, UML Architecture, and Tech Stack."""
        return render_template('about.html', active_page='about')

    @app.route('/report/<int:run_id>')
    def download_report(run_id):
        """Generates and serves authentic ReportLab PDF report for specified run."""
        run = db.get_run(run_id)
        if not run:
            abort(404, description=f"Analysis Run #{run_id} not found.")

        try:
            from services.report_generator import ReportGenerator
            generator = ReportGenerator(db)

            # Ensure clean regeneration by removing any stale cached file
            rep_folder = app.config.get('REPORT_FOLDER', Config.REPORT_FOLDER)
            pdf_filename = f"MarineVision_Analysis_{run_id}.pdf"
            cached_path = os.path.join(rep_folder, pdf_filename)
            if os.path.exists(cached_path):
                try:
                    os.remove(cached_path)
                except Exception:
                    pass

            pdf_path = generator.generate_report(run_id)
            if not os.path.exists(pdf_path):
                abort(500, description="Failed to create PDF report file on server.")

            download_name = f"MarineVision_Analysis_{run_id}.pdf"
            response = send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=download_name,
                max_age=0
            )
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        except Exception as e:
            flash(f"Error generating PDF report: {str(e)}", "danger")
            return redirect(url_for('analyze_view', image_id=run['image_id']))

    # =========================================================================
    # Safe Static / Media File Serving
    # =========================================================================

    @app.route('/uploads/<path:filename>')
    def serve_upload(filename):
        """Safely serves uploaded sonar images."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/processed/<path:filename>')
    def serve_processed(filename):
        """Safely serves preprocessed / annotated sonar images."""
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

    # =========================================================================
    # Error Handlers
    # =========================================================================

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html', error=e), 404

    @app.errorhandler(413)
    def request_entity_too_large(e):
        flash("Uploaded file exceeds the maximum 10 MB limit.", "danger")
        return redirect(url_for('analyze'))

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('404.html', error=e, is_500=True), 500

    return app

app = create_app()

if __name__ == '__main__':
    # Local development server execution
    # use_reloader=False prevents double model loading (PyTorch + YOLO) which causes OOM crashes
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False, threaded=True)
