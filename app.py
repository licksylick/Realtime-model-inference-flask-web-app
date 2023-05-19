from flask import Flask, flash, request, render_template, Response
import sys
from pathlib import Path
from config import UPLOAD_FOLDER, REMOVE_TIME, GPU_NUM, VIDEO_RESOLUTION, PORT
from utils import remove_old_files, model_predictions, draw_predictions
import cv2
import os
import multiprocessing as mp
import warnings


warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'


action_event = mp.Event()
name_to_save = ''
stop_event = mp.Event()


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:
    pass


@app.route('/')
def upload_form():
    # global stop_event
    # if not stop_event.is_set():
    #     stop_event.set()
    return render_template('home.html')


def webcam_capture(frame_queue, detection_queue):
    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_RESOLUTION[1])
    global stop_event

    while True:
        global action_event, name_to_save
        if stop_event.is_set():
            break
        ret, frame = camera.read()
        if not ret:
            break
        else:
            if frame_queue.empty():
                frame_queue.put(frame)
            if detection_queue.empty():
                continue

        if action_event.is_set():  # if save_event TRUE
            img_name = name_to_save + '.jpg'
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, img_name), frame)
            action_event.clear()

        if detection_queue.empty():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        boxes, names = detection_queue.get()

        draw_predictions(frame, boxes, names)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    stop_event.clear()


def process_frames(frame_queue, detection_queue):
    global stop_event
    while True:
        if stop_event.is_set():
            break
        if frame_queue.empty():
            continue
        frame = frame_queue.get()

        # TODO edit model_predictions in utils.py
        boxes, names = model_predictions(frame)
        detection_queue.put((boxes, names))


@app.route('/videoo', methods=["GET", "POST"])
def videoo():
    return render_template('videoo.html')


@app.route('/requests_task', methods=["GET", "POST"])
def tasks():
    global action_event, name_to_save
    if request.method == 'POST':
        input_text = request.form['text']
        name_to_save = input_text
        print(f'Saving: {name_to_save}')
        action_event.set()  # TRUE
        return videoo()
    elif request.method == 'GET':
        return videoo()

    return videoo()



@app.route('/video')
def video():
    remove_old_files(UPLOAD_FOLDER, REMOVE_TIME)

    global stop_event, action_event
    stop_event.clear()

    frame_queue = mp.Queue()
    detection_queue = mp.Queue()

    p1 = mp.Process(target=webcam_capture, args=(frame_queue, detection_queue))

    p2 = mp.Process(target=process_frames, args=(frame_queue, detection_queue))

    p1.start()
    p2.start()

    mp.freeze_support()

    return Response(webcam_capture(frame_queue, detection_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=PORT)