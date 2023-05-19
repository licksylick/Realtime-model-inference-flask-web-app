import os
import time
import cv2


def draw_predictions(frame, boxes, names):
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return frame


# TODO replace according to your requirements (model applying)
def model_predictions(frame):
    names = ['Person']
    boxes = [[20, 100, 200, 20]]
    return boxes, names


def remove_old_files(path, last_modified_time):
    for f in os.listdir(path):
        if os.path.isfile(f):
            if os.stat(os.path.join(path, f)).st_mtime < time.time() - last_modified_time:
                os.remove(os.path.join(path, f))


def get_index_of_max_box(boxes):
    largest_area = 0
    largest_area_index = 0

    for i, (top, right, bottom, left) in enumerate(boxes):
        area = (right - left) * (bottom - top)
        if area > largest_area:
            largest_area = area
            largest_area_index = i

    return largest_area_index

