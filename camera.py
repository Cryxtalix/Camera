import cv2 as cv
import math
import pandas
import time

class Camera:
        def __init__(self, src) -> None:
                self.video = cv.VideoCapture(src)
                if not self.video.isOpened():
                        raise Exception("Cannot open camera")
                self.start_time = 0
                self.stream_time = 0
                self.frame_count: int = 0
                self.running: bool = False

                # Initial settings
                self.project_name: str = ""
                self.record_all: bool = False
                self.record_motion: bool = False

                # Settings
                self.show_motion_detection: bool = False #Show motion areas
                self.edge_detection: bool = False
                self.facial_recognition: bool = False

                # Detailed settings

                self.background_frame = 0

        def halt(self) -> None:
                self.running = 0
                self.video.release()

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
                        self.frame = self.manipulate(self.frame)

                        # Encode and return video stream
                        ret, self.encoded_frame = cv.imencode('.jpg', self.frame)
                        self.byte_encoded_frame = self.encoded_frame.tobytes()
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + self.byte_encoded_frame + b'\r\n')

        def initial_settings(self, category: str, state) -> None:
                """
                * Sets initial video options, cannot use when camera is already operational
                * Category: "record_all", "record_motion", "input_name"
                """
                if not self.running:
                        if category == "record_all" and type(state) == bool:
                                self.record_all = state
                                return
                        if category == "record_motion" and type(state) == bool:
                                self.record_motion = state
                                return
                        if category == "input_name" and type(state) == str:
                                self.project_name = state
                        else:
                                raise Exception("Invalid entry")

        def update_settings(self, category: str, state: bool) -> None:
                """
                * Changes video options while camera is operational
                * Category: "show_motion_detection", "facial_recognition"
                * State: True or False
                """
                if state != True and state != False:
                        return
                if category == "show_motion_detection":
                        self.show_motion_detection = state
                        return
                elif category == "edge_detection":
                        self.edge_detection = state
                        return
                elif category == "facial_recognition":
                        self.facial_recognition = state
                        return
                elif category == "smoothing_value":
                        self.smoothing_kernel = state
                else:
                        raise Exception("Please select from available categories")
                
        def updateBackground(self) -> None:
                """
                As the first frame captured is used as the background frame by default, this function
                updates the background frame if necessary.
                """
                # Convert the frames to grayscale
                grayscale = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                # Gaussian blur for smoothing and reducing noise
                grayscale = cv.GaussianBlur(grayscale, (21, 21), 0)
                self.background_frame = grayscale
        
        def manipulate(self, frame):
                # Convert the frames to grayscale
                grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Gaussian blur for smoothing and reducing noise
                gray_blur = cv.GaussianBlur(grayscale, (21, 21), 0)
                # Save the first frame used for computation
                if self.frame_count == 1:
                        self.background_frame = gray_blur
                
                if self.edge_detection:
                        out_frame = cv.Canny(grayscale, 100, 200)

                elif self.facial_recognition:
                        out_frame = frame
                
                else:
                        out_frame = frame
                
                # Perform motion detection
                subtracted_frame = cv.absdiff(self.background_frame, gray_blur)
                thresh_frame = cv.threshold(subtracted_frame, 30, 255, cv.THRESH_BINARY)[1]
                thresh_frame = cv.dilate(thresh_frame, None, iterations = 2)
                contours,_ = cv.findContours(thresh_frame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                        if cv.contourArea(contour) < 10000:
                                continue
                        
                        if self.show_motion_detection:
                                (x, y, w, h) = cv.boundingRect(contour)
                                out_frame = cv.rectangle(out_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                return out_frame

