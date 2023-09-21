import camera
from flask import Flask, request, Response, render_template, redirect, url_for

app = Flask(__name__)
mycamera = camera.Camera(0)

# Index.html
@app.route('/', methods=["GET","POST"])
def index():
        if request.method == "POST":
                if request.form.get("input_name"):
                        mycamera.initial_settings("input_name", request.form['input_name'])

                if request.form.get('checkbox1') == "on":
                        mycamera.initial_settings("record_all", True)

                if request.form.get('checkbox2') == "on":
                        mycamera.initial_settings("record_motion", True)
                return redirect(url_for("video"))
        
        return render_template("index.html")

# video.html
@app.route("/video/", methods=["GET","POST"])
def video():
        # Initial values
        exposure_value: int = 50
        show_motion_detection: str = "off"
        facial_recognition: str = "off"
        edge_detection: str = "off"

        if request.method == "POST":
                # Gets exposure value
                exposure_value = int(request.form["exposure_slider"])

                # Toggle display motion detection on/off
                if request.form.get("show_motion_detection") == "Toggle":
                        mycamera.update_settings("show_motion_detection", not mycamera.show_motion_detection)

                # Updates background frame
                if request.form.get("update_background") == "Update Background":
                        mycamera.updateBackground()

                # Toggles edge detection on/off
                if request.form.get("edge_detection") == "Toggle":
                        mycamera.update_settings("edge_detection", not mycamera.edge_detection)
                        mycamera.update_settings("facial_recognition", False)

                # Toggles facial recognition on/off
                if request.form.get("facial_recognition") == "Toggle":
                        mycamera.update_settings("facial_recognition", not mycamera.facial_recognition)
                        mycamera.update_settings("edge_detection", False)

                if request.form.get("third") == "third":
                        print(mycamera.show_motion_detection)
                        print(mycamera.project_name)
                        print(mycamera.stream_time)
        
        show_motion_detection = "on" if mycamera.show_motion_detection == True else "off"
        facial_recognition = "on" if mycamera.facial_recognition == True else "off"
        edge_detection = "on" if mycamera.edge_detection == True else "off"

        return render_template("video.html", 
                               exposure_value=exposure_value, 
                               show_motion_detection=show_motion_detection, 
                               edge_detection=edge_detection,
                               facial_recognition=facial_recognition)

@app.route("/video_feed/")
def video_feed():
        return Response(mycamera.run_camera(), mimetype= "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
        app.run()