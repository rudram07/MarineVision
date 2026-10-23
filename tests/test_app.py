import os
import io
import pytest
from app import create_app
from config import Config
from database import DatabaseManager

class TestConfig(Config):
    TESTING = True
    INSTANCE_FOLDER = os.path.join(Config.BASE_DIR, 'instance_test')
    DATABASE_PATH = os.path.join(INSTANCE_FOLDER, 'marinevision_test.db')
    UPLOAD_FOLDER = os.path.join(Config.BASE_DIR, 'uploads_test')
    PROCESSED_FOLDER = os.path.join(Config.BASE_DIR, 'processed_test')
    REPORT_FOLDER = os.path.join(Config.BASE_DIR, 'reports_test')

@pytest.fixture
def client():
    # Setup test app
    app = create_app(TestConfig)
    with app.test_client() as test_client:
        with app.app_context():
            db = DatabaseManager(TestConfig.DATABASE_PATH, TestConfig.SCHEMA_PATH)
            db.init_db()
        yield test_client

    # Cleanup test dirs after fixture
    import shutil
    for folder in [TestConfig.INSTANCE_FOLDER, TestConfig.UPLOAD_FOLDER, TestConfig.PROCESSED_FOLDER, TestConfig.REPORT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

def test_five_main_page_routes(client):
    """Verifies all five required main navigation pages return HTTP 200."""
    routes = ['/intro', '/', '/analyze', '/history', '/about']
    for route in routes:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} failed with status {response.status_code}"

def test_navigation_sidebar_present(client):
    """Verifies the persistent 5-page sidebar navigation is rendered."""
    response = client.get('/')
    html = response.data.decode('utf-8')
    assert 'nav-intro' in html
    assert 'nav-dashboard' in html
    assert 'nav-analyze' in html
    assert 'nav-history' in html
    assert 'nav-about' in html

def test_valid_image_upload_and_pipeline(client):
    """Tests valid sonar image upload, OpenCV preprocessing, and SQLite persistence."""
    import cv2
    import numpy as np

    # Generate synthetic in-memory test sonar image
    test_img = np.ones((200, 300), dtype=np.uint8) * 128
    _, encoded = cv2.imencode('.png', test_img)
    image_bytes = encoded.tobytes()

    data = {
        'image': (io.BytesIO(image_bytes), 'sonar_seabed_test.png')
    }

    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode('utf-8')

    # Verify combined page displays results
    assert 'sonar_seabed_test.png' in html
    assert 'PREPROCESSED / CLAHE ENHANCED' in html
    assert 'Download Analysis Report (PDF)' in html
    assert 'Marine debris detection is pending model integration. No AI detections have been generated.' in html

def test_invalid_extension_upload(client):
    """Rejects disallowed extensions like .txt or .exe."""
    data = {
        'image': (io.BytesIO(b"dummy text content"), 'malicious.txt')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    html = response.data.decode('utf-8')
    assert 'Unsupported file format' in html

def test_empty_file_upload(client):
    """Rejects 0-byte uploads."""
    data = {
        'image': (io.BytesIO(b""), 'empty.png')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    html = response.data.decode('utf-8')
    assert 'Uploaded file is empty' in html

def test_corrupted_image_upload(client):
    """Rejects corrupted file content claiming to be an image."""
    data = {
        'image': (io.BytesIO(b"NOT_A_REAL_IMAGE_DATA_CORRUPTED"), 'corrupted.png')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    html = response.data.decode('utf-8')
    assert 'Corrupted or invalid image content' in html

def test_sample_sonar_loader(client):
    """Tests 1-click synthetic sample loading and processing."""
    response = client.post('/api/load-sample/debris', headers={'Accept': 'application/json'})
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] is True
    assert 'image_id' in json_data

    # Check results view
    res_view = client.get(f"/analyze/{json_data['image_id']}")
    assert res_view.status_code == 200
    assert 'Download Analysis Report (PDF)' in res_view.data.decode('utf-8')

def test_pdf_report_download(client):
    """Verifies real ReportLab PDF report generation and download."""
    # First load sample to ensure run exists
    load_resp = client.post('/api/load-sample/anomaly', headers={'Accept': 'application/json'})
    image_id = load_resp.get_json()['image_id']

    db = DatabaseManager(TestConfig.DATABASE_PATH)
    latest_run = db.get_latest_run_for_image(image_id)
    assert latest_run is not None

    run_id = latest_run['run_id']
    report_resp = client.get(f'/report/{run_id}')
    assert report_resp.status_code == 200
    assert report_resp.mimetype == 'application/pdf'
    assert report_resp.data.startswith(b'%PDF')
    assert len(report_resp.data) > 1000

def test_history_search_and_filter(client):
    """Verifies analysis history displays real records and supports search/filter."""
    client.post('/api/load-sample/debris', headers={'Accept': 'application/json'})
    
    resp = client.get('/history')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert 'sonar_debris_sample.png' in html
    assert 'COMPLETED' in html

    # Test search
    search_resp = client.get('/history?q=debris')
    assert search_resp.status_code == 200
    assert 'sonar_debris_sample.png' in search_resp.data.decode('utf-8')
