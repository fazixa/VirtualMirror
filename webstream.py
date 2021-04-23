# import the necessary packages
import argparse
import numpy as np
import imutils
import time
import cv2
import dlib
from makeup.eyeshadow import eyeshadow
from flask import Flask, render_template, url_for, request, Response



outputFrame = None
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")


def detect_motion():
    
    global cap, outputFrame, prev, frame_rate

    while True:
        ret, frame = cap.read()
        time_elapsed = time.time() - prev
        frame = imutils.resize(frame, width=650)

        if(time_elapsed > 1./frame_rate):
            prev = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            eye = eyeshadow(frame2)
            detected_faces = detector(gray, 0)
            landmarks_x = []
            landmarks_y = []    
            try:

                pose_landmarks = face_pose_predictor(gray, detected_faces[0])

                for i in range(68):
                    landmarks_x.append(pose_landmarks.part(i).x)
                    landmarks_y.append(pose_landmarks.part(i).y)
                frame = eye.apply_eyeshadow(landmarks_x,landmarks_y,r,g,b,0.5)
                outputFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(e)




def generate():
    # grab global references 
    global outputFrame
    # loop over frames from the output stream
    while True:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')



app = Flask(__name__)
            
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/', methods=['POST'])
def my_form_post():
    return render_template("index.html" )

@app.route('/process', methods=['POST'])
def process():
    global r,g,b
    r = float(request.form['r'])
    g = float(request.form['g'])
    b = float(request.form['b'])
    print("hi")
 
    return "ok"

@app.route('/caprelease', methods=['POST'])
def caprelease():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    return "cap released"

@app.route('/opencam', methods=['POST'])
def opencam():
    global cap, prev, frame_rate, r, g, b
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    frame_rate = 15
    prev = 0
    r = 100
    g = 20
    b = 90

    detect_motion()
    return "cap opened"



# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())


    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
cap.stop()