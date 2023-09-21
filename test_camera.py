import unittest
import cv2 as cv
import camera

class TestCamera(unittest.TestCase):
        def testMainExists(self):
                self.assertIsNotNone(camera.Camera)

        def testInitial(self):
                cam = camera.Camera(0)
                self.assertEqual(cam.frame_count, 0)
                self.assertEqual(cam.stream_time, 0)
                self.assertEqual(cam.running, 0)
                self.assertEqual(cam.record_all, 0)
                self.assertEqual(cam.record_motion, 0)
                self.assertEqual(cam.motion_detection, 0)
                self.assertEqual(cam.facial_recognition, 0)
                self.assertIsNone(cam.background_frame)

        def testUpdateInitial(self):
                cam = camera.Camera(0)
                cam.initial_settings("record_all", 1)
                cam.initial_settings("record_motion", 1)
                self.assertEqual(cam.record_all, 1)
                self.assertEqual(cam.record_motion, 1)

        def testUpdateInitialJunkInput(self):
                cam = camera.Camera(0)
                cam.initial_settings("record_all", 0)
                cam.initial_settings("record_motion", 0)
                self.assertEqual(cam.record_all, 0)
                self.assertEqual(cam.record_motion, 0)

        def testUpdateInitialIfRunning(self):
                cam = camera.Camera(0)
                cam.run_camera()
                cam.initial_settings("record_all", 1)
                self.assertEqual(cam.running, 1)
                
unittest.main()