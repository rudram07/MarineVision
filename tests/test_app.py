import os
import io
import shutil
import unittest
import cv2
import numpy as np

from app import create_app
from config import Config
from database import DatabaseManager

class TestConfig(Config):
    TESTING = True
    INSTANCE_FOLDER = os.path.join(Config.BASE_DIR, 'instance_test_app')
    DATABASE_PATH = os.path.join(INSTANCE_FOLDER, 'marinevision_test.db')
    UPLOAD_FOLDER = os.path.join(Config.BASE_DIR, 'uploads_test_app')
    PROCESSED_FOLDER = os.path.join(Config.BASE_DIR, 'processed_test_app')
    REPORT_FOLDER = os.path.join(Config.BASE_DIR, 'reports_test_app')

class AppIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()
        with self.app.app_context():
            self.db = DatabaseManager(TestConfig.DATABASE_PATH, TestConfig.SCHEMA_PATH)
            self.db.init_db()

    def tearDown(self):
        for folder in [TestConfig.INSTANCE_FOLDER, TestConfig.UPLOAD_FOLDER, TestConfig.PROCESSED_FOLDER, TestConfig.REPORT_FOLDER]:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)

    def test_five_main_page_routes(self):
        """Verifies all five required main navigation pages return HTTP 200."""
        routes = ['/intro', '/', '/analyze', '/history', '/about']
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with status {response.status_code}")

    def test_navigation_sidebar_present(self):
        """Verifies the persistent top navigation is rendered with Introduction link removed."""
        response = self.client.get('/')
        html = response.data.decode('utf-8')
        self.assertNotIn('nav-intro', html)
        self.assertIn('nav-dashboard', html)
        self.assertIn('nav-analyze', html)
        self.assertIn('nav-history', html)
        self.assertNotIn('nav-about', html)
        self.assertIn('footer-about', html)
        self.assertIn('About System', html)

    def test_valid_image_upload_and_pipeline(self):
        """Tests valid sonar image upload, OpenCV preprocessing, and SQLite persistence."""
        test_img = np.ones((200, 300), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.png', test_img)
        image_bytes = encoded.tobytes()

        data = {
            'image': (io.BytesIO(image_bytes), 'sonar_seabed_test.png')
        }

        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')

        self.assertIn('sonar_seabed_test.png', html)
        self.assertIn('PREPROCESSED / CLAHE ENHANCED', html)
        self.assertIn('Download Analysis Report (PDF)', html)

    def test_invalid_extension_upload(self):
        """Rejects disallowed extensions like .txt or .exe."""
        data = {
            'image': (io.BytesIO(b"dummy text content"), 'malicious.txt')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        html = response.data.decode('utf-8')
        self.assertIn('Unsupported file format', html)

    def test_empty_file_upload(self):
        """Rejects 0-byte uploads."""
        data = {
            'image': (io.BytesIO(b""), 'empty.png')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        html = response.data.decode('utf-8')
        self.assertIn('Uploaded file is empty', html)

    def test_corrupted_image_upload(self):
        """Rejects corrupted file content claiming to be an image."""
        data = {
            'image': (io.BytesIO(b"NOT_A_REAL_IMAGE_DATA_CORRUPTED"), 'corrupted.png')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        html = response.data.decode('utf-8')
        self.assertIn('Corrupted or invalid image content', html)

    def test_sample_sonar_loader(self):
        """Tests 1-click synthetic sample loading and processing."""
        response = self.client.post('/api/load-sample/debris', headers={'Accept': 'application/json'})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data['success'])
        self.assertIn('image_id', json_data)

        # Check results view
        res_view = self.client.get(f"/analyze/{json_data['image_id']}")
        self.assertEqual(res_view.status_code, 200)
        self.assertIn('Download Analysis Report (PDF)', res_view.data.decode('utf-8'))

    def test_pdf_report_download(self):
        """Verifies real ReportLab PDF report generation and download."""
        load_resp = self.client.post('/api/load-sample/anomaly', headers={'Accept': 'application/json'})
        image_id = load_resp.get_json()['image_id']

        with self.app.app_context():
            db = DatabaseManager(TestConfig.DATABASE_PATH)
            latest_run = db.get_latest_run_for_image(image_id)
            self.assertIsNotNone(latest_run)

            run_id = latest_run['run_id']
            report_resp = self.client.get(f'/report/{run_id}')
            self.assertEqual(report_resp.status_code, 200)
            self.assertEqual(report_resp.mimetype, 'application/pdf')
            self.assertTrue(report_resp.data.startswith(b'%PDF'))
            self.assertGreater(len(report_resp.data), 1000)

    def test_history_search_and_filter(self):
        """Verifies analysis history displays real records and supports search/filter."""
        self.client.post('/api/load-sample/debris', headers={'Accept': 'application/json'})
        
        resp = self.client.get('/history')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn('sonar_debris_sample.png', html)
        self.assertIn('COMPLETED', html)

        search_resp = self.client.get('/history?q=debris')
        self.assertEqual(search_resp.status_code, 200)
        self.assertIn('sonar_debris_sample.png', search_resp.data.decode('utf-8'))

if __name__ == '__main__':
    unittest.main()
