"""
 * @author Faezeh Shayesteh
 * @email shayesteh.fa@gmail.com
 * @create date 2021-04-23 21:26:28
 * @modify date 2021-04-23 21:26:28
 * @desc [description]
"""

import imutils
import time
import dlib
import cv2
import numpy as np
from makeup.eyeshadow import eyeshadow
from makeup.blush import blush
import threading


def eyeshadow_worker(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity, out_queue) -> None:
    eyes = eyeshadow(w_frame)
    result = eyes.apply_eyeshadow(w_landmarks_x, w_landmarks_y, r, g, b, intensity)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (eyes.x_all, eyes.y_all)
    })


def blush_worker(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity, out_queue) -> None:
    cheeks = blush(w_frame)
    result = cheeks.apply_blush(w_landmarks_x, w_landmarks_y, r, g, b, intensity)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (cheeks.x_all, cheeks.y_all)
    })


makeup_workers = [eyeshadow_worker, blush_worker]


def join_makeup_workers(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity):
    processes = []
    shared_queue = []

    for worker in makeup_workers:
        p = threading.Thread(target=worker, args=(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity, shared_queue), daemon=True)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    final_image = shared_queue.pop()['image']

    while len(shared_queue) > 0:
        temp_img = shared_queue.pop()
        (range_x, range_y) = temp_img['range']
        temp_img = temp_img['image']

        for x, y in zip(range_x, range_y):
            final_image[x, y] = temp_img[x, y]
        # final_image[range_x, range_y] = temp_imgimg[range_x, range_y]

    return final_image


if __name__ == '__main__':
    # initiating camera
    prev = 0
    frame_rate = 15
    detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(2)
    time.sleep(2.0)

    # applying makeup on frames
    while True:
        # frame rate and resize frame
        ret, frame = cap.read()
        time_elapsed: float = time.time() - prev
        frame = imutils.resize(frame, width=700)

        if time_elapsed > 1. / frame_rate:
            # preparing frame
            prev = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # eye = eyeshadow(frame2)
            # cheek = blush(frame2)
            # detect faces in frame
            detected_faces = detector(gray, 0)
            landmarks_x = []
            landmarks_y = []
            # get landmarks of the face
            try:
                pose_landmarks = face_pose_predictor(gray, detected_faces[0])
                for i in range(68):
                    landmarks_x.append(pose_landmarks.part(i).x)
                    landmarks_y.append(pose_landmarks.part(i).y)
                # frame = cheek.apply_blush(landmarks_x, landmarks_y, 100, 20, 90, 0.5)
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = join_makeup_workers(frame2, landmarks_x, landmarks_y, 100, 20, 90, 0.5)
            except Exception as e:
                print(e)
        # show face with applied makeup
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
