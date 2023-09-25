import cv2 as cv
import math
import pandas as pd
import time
from datetime import datetime
import numpy as np
import pygame

class Camera:
        def __init__(self, src) -> None:
                self.video = cv.VideoCapture(src)
                if not self.video.isOpened():
                        raise Exception("Cannot open camera")
                self.start_time = 0
                self.stream_time = 0
                self.frame_count: int = 0
                self.running: bool = False
                self.background_frame = 0

                # Initial settings
                self.project_name: str = ""

                # Settings
                self.motion_detection: bool = False
                self.show_motion_detection: bool = False # Draw on video
                self.facial_recognition: bool = False
                self.show_facial_recognition: bool = False # Draw on video
                self.edge_detection: bool = False
                self.exposure: int = 0

                self.face_classifier = cv.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")
                pygame.init()
                pygame.mixer.music.load("./assets/278142__ricemaster__effect_notify.wav")

                if not self.video.set(cv.CAP_PROP_AUTO_EXPOSURE, 0):
                        self.exposure_possible = False
                else:
                        self.exposure_possible = True

                # Detection settings
                self.motion_detected: bool = False
                self.face_detected: bool = False
                #self.data = pd.DataFrame(columns=['Frame count', 'DateTime', 'Motion detected', 'Face detected'])
                self.log_data = []

        def halt(self) -> None:
                self.running = False
                self.video.release()
                cv.destroyAllWindows()

                df = pd.DataFrame(self.log_data)
                df.to_csv(f"./csv_out/project_{self.project_name}.csv")

        def run_camera(self) -> None:
                self.running = 1
                if self.start_time == 0: # First run of camera
                        self.start_time = time.time()

                while self.running:
                        self.success, self.frame = self.video.read()
                        if not self.success:
                                raise Exception("Cannot read video")
                        self.frame_count += 1
                        self.stream_time = math.floor(time.time() - self.start_time)

                        # Perform image manipulation
                        output_frame = self.manipulate(self.frame)

                        # Encode and return video stream
                        ret, self.encoded_frame = cv.imencode('.jpg', output_frame)
                        self.byte_encoded_frame = self.encoded_frame.tobytes()
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + self.byte_encoded_frame + b'\r\n')

                        # Update dataframe
                        now = datetime.now()
                        dt_str = now.strftime("%d/%m/%Y %H:%M:%S")
                        #self.data.loc[len(self.data.index)] = [self.frame_count, dt_str, self.motion_detected, self.face_detected]
                        self.log_data.append({"Framecount": self.frame_count,
                                              "Timestamp": dt_str,
                                              "Motion Detected": self.motion_detected,
                                              "Face Detected": self.face_detected})

        def initial_settings(self, category: str, state) -> None:
                """
                * Sets initial video options, cannot use when camera is already operational
                """
                if not self.running:
                        if category == "record_all" and type(state) == bool:
                                self.record_all = state
                                return
                        if category == "record_motion" and type(state) == bool:
                                self.record_motion = state
                                return
                        if category == "project_name" and type(state) == str:
                                self.project_name = state
                        else:
                                raise Exception("Invalid entry")
                        
        def set_exposure(self, exposure):
                if self.exposure_possible:
                        self.video.set(cv.CAP_PROP_EXPOSURE, exposure)

        def update_settings(self, category: str, state: bool) -> None:
                """
                * Changes video options while camera is operational
                * State: True or False
                """
                if state != True and state != False:
                        return
                if category == "motion_detection":
                        self.motion_detection = state
                        return
                elif category == "show_motion_detection":
                        self.show_motion_detection = state
                        return
                elif category == "edge_detection":
                        self.edge_detection = state
                        return
                elif category == "facial_recognition":
                        self.facial_recognition = state
                        return
                elif category == "show_facial_recognition":
                        self.show_facial_recognition = state
                        return
                elif category == "exposure":
                        self.set_exposure(state)
                else:
                        raise Exception("Please select from available categories")
                
        def updateBackground(self) -> None:
                """
                As the first frame captured is used as the background frame by default, 
                call to choose current frame as the new background frame
                """
                # Convert the frames to grayscale
                grayscale = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                # Gaussian blur for smoothing and reducing noise
                grayscale = cv.GaussianBlur(grayscale, (21, 21), 0)
                self.background_frame = grayscale
        
        def manipulate(self, frame):
                """
                Private function, performs image manipulation.

                Do not access directly.
                """
                # Convert the frames to grayscale
                grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Gaussian blur for smoothing and reducing noise
                gray_blur = cv.GaussianBlur(grayscale, (21, 21), 0)
                # Save the first frame used for computation
                if self.frame_count == 1:
                        self.background_frame = gray_blur

                self.motion_detected = 0
                self.face_detected = 0

                # Frame manipulation
                frame = self.algo_edge_detection(grayscale, frame)
                
                frame = self.algo_motion_detection(grayscale, frame)
                if self.motion_detected:
                        pygame.mixer.music.play()

                frame = self.algo_facial_detection(grayscale, frame)

                return frame

        def algo_motion_detection(self, grayscale, frame):
                if self.motion_detection:
                        subtracted_frame = cv.absdiff(self.background_frame, grayscale)
                        thresh_frame = cv.threshold(subtracted_frame, 30, 255, cv.THRESH_BINARY)[1]
                        thresh_frame = cv.dilate(thresh_frame, None, iterations = 2)
                        contours,_ = cv.findContours(thresh_frame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                                if cv.contourArea(contour) < 10000:
                                        continue
                                
                                # Motion has been detected
                                self.motion_detected = True
                                # Display bounding rect (optional)
                                if self.show_motion_detection:
                                        (x, y, w, h) = cv.boundingRect(contour)
                                        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                return frame
                
        def algo_facial_detection(self, grayscale, frame):
                if self.facial_recognition:
                        faces = self.face_classifier.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=3)
                        if len(faces) > 0:
                                # Face detected
                                self.face_detected = True
                                pass
                        if self.show_facial_recognition:
                                for (x, y, w, h) in faces:
                                        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                                
                return frame
                        
        def algo_edge_detection(self, grayscale, frame):
                if self.edge_detection:
                        # Sorbel filter kernel
                        edge_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                        edge_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

                        convolve_y = cv.filter2D(grayscale, -1, edge_kernel_y)
                        convolve_x = cv.filter2D(grayscale, -1, edge_kernel_x)

                        frame = convolve_x + convolve_y

                return frame