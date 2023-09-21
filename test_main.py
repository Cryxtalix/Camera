import unittest
from main import app

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def tearDown(self):
        pass

    def testIndexGET(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200) # 200 ok
        self.assertIn(b'Record entire stream', response.data)

    def testIndexPOST(self):
        response = self.app.post('/', data={'checkbox1': 'on', 'checkbox2': 'on'})
        # Test redirect
        self.assertEqual(response.status_code, 302) # 302 found
        self.assertIn(b'video', response.data)

    def testVideoGET(self):
        response = self.app.get('/video/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Motion detection', response.data)

    def testVideoPOST(self):
        response = self.app.post('/video/', data={'exposure_slider': '75'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'75', response.data)

    def testVideoFeed(self):
        response = self.app.get('/video_feed/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    unittest.main()
