import cv2
import copy
import time
import os, sys
import numpy as np
import tensorflow as tf
import threading
from queue import Queue
from scipy.special import softmax, expit
from Model.utils import color_
from Model.infer_gpu import get_anchors, nms_oneclass, decode_bbox, decode_landm


def model_predict(interpreter, input, output, img):

    # print(input, '\n', img.shape)
    interpreter.set_tensor(input[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output[0]['index'])

    return output


def get_fram():
    global fram_queue, capture, global_flag
    while(capture.isOpened() and global_flag):
        ret, img = capture.read()
        if(ret):
            fram_queue.put(img)


def image_process(img):

    img = cv2.copyMakeBorder(img, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.pad(img, ((0,0), (80, 80), (0,0), (0,0)), constant_values=0)
    det_img = ((img / 255. - 0.5) / 1)[None, ...]
    det_img = det_img.astype(np.float32)
    # print(np.max(det_img), np.min(det_img))
    return det_img


def get_batch_pic():
    global fram_queue, batch_queue, global_flag
    while(global_flag):
        if(fram_queue.qsize() >= 10):
            # print('[INFO]>>> 开始批处理')
            frams = [fram_queue.get() for _ in range(1)]
            frams = list(map(lambda x: np.expand_dims(x, axis=0), frams))
            batch_image = np.concatenate([i for i in frams], axis=0)
            images = image_process(batch_image)
            batch_queue.put(images)


def parse_predict(predictions):
    global anchors
    variances = [0.1, 0.2]
    obj_thresh = 0.7
    nms_threshold = 0.4
    """ parser
        bbox: [0, 1, 2, 33], landm: [4 -> 13], clses: [14, 15]
    """
    bbox, landm, clses = np.split(predictions[0], [4, -2], 1)
    """ softmax class"""
    clses = softmax(clses, -1)
    score = clses[:, 1]
    """ decode """
    bbox = decode_bbox(bbox, anchors, variances)
    bbox = bbox * np.tile([640, 640], [2])
    """ filter low score """
    inds = np.where(score > obj_thresh)[0]
    # print('inds.shape:{}'.format(inds.shape))
    bbox = bbox[inds]
    score = score[inds]
    """ keep top-k before NMS """
    order = np.argsort(score)[::-1]
    bbox = bbox[order]
    # score = score[order]
    """ do nms """
    keep = nms_oneclass(bbox, score, nms_threshold)
    # keep = 0
    # print('score.shape:{}'.format(score.shape))
    # print('bbox.shape:{}'.format(bbox.shape))

    return [bbox, keep]


def get_crop_img(draw_img, bbox, keep):
    for b in bbox[keep].astype(int):
        # b[:2]--> 矩形框左上角, b[2:]--> 矩形框右下角
        cv2.rectangle(draw_img, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 2)
        # (cx, cy) 矩形框中心坐标
        cx, cy = (b[:2] + b[2:]) // 2
        # 矩形框的1/2边长
        halfw = np.max(b[2:] - b[:2]) // 2
        # 重新调整矩形框
        croped_img: np.ndarray = draw_img[cy - halfw:cy + halfw, cx - halfw:cx + halfw]
        # 调整框的wh
        croped_wh = croped_img.shape[1::-1]
        s_point = np.array([cx - halfw, cy - halfw])

        if croped_wh[0] == croped_wh[1] and min(croped_wh) > 10:
            croped_img = cv2.resize(croped_img, (112, 112))
            croped_img = ((croped_img / 255. - 0.5) / 1)[None, ...]
            croped_img = croped_img.astype(np.float32)
            return (croped_img, croped_wh, s_point)
        else:
            return None




def detect_face(ids):
    global origin_queue, fram_queue
    global global_flag
    interpreter_retinaface = tf.lite.Interpreter(model_path='models/tflite/retinaface.tflite')
    interpreter_retinaface.allocate_tensors()
    retina_input_details = interpreter_retinaface.get_input_details()
    retina_output_details = interpreter_retinaface.get_output_details()

    while (global_flag):
        if(fram_queue.qsize() >= 1):

            imgs = fram_queue.get()
            img = copy.deepcopy(imgs)
            imgs = image_process(imgs)
            bt = time.time()
            predictions = model_predict(interpreter_retinaface, retina_input_details, retina_output_details, imgs)
            print('[INFO]>>> retina{}处理一帧:{}/s'.format(ids, time.time() - bt))
            # print(predictions[0].shape)
            predictions = parse_predict(predictions)
            predictions.append(img)


            origin_queue.put(predictions)



def landmark_face(ids):

    global origin_queue, randmark_queue
    global global_flag
    interpreter_pfld = tf.lite.Interpreter(model_path='models/tflite/pfld_infer.tflite')
    interpreter_pfld.allocate_tensors()
    pfld_input_details = interpreter_pfld.get_input_details()
    pfld_output_details = interpreter_pfld.get_output_details()

    while(global_flag):
        if(origin_queue.qsize() >= 1):

            bbox, keep, draw_img = origin_queue.get()
            crops_info = get_crop_img(draw_img, bbox, keep)
            if(crops_info is None):
                pass
            else:
                croped_img, croped_wh, s_point = crops_info
                # print('croped_img.shape:{}, croped_wh:{} s_points:{}'.format(croped_img.shape, croped_wh, s_point))
                bt = time.time()
                landmarks = model_predict(interpreter_pfld, pfld_input_details, pfld_output_details, croped_img)
                print('[INFO]>>> pfld{}处理一帧时间:{}/s'.format(ids,time.time() - bt))
                for i, landm in enumerate(np.reshape(expit(landmarks), (-1, 2)) * croped_wh):
                    color = color_(i)
                    cv2.circle(draw_img, tuple((s_point + landm).astype(int)), 1, color)
            randmark_queue.put(draw_img)




def show():
    global randmark_queue
    global capture
    global global_flag
    while(True):
        if(randmark_queue.qsize() >= 1):
            img = randmark_queue.get()
            # time.sleep(0.05)
            cv2.imshow('fram', img)
            if cv2.waitKey(1) == ord('q'):
                global_flag = False
                break
    capture.release()
    cv2.destroyAllWindows()


def camera_set():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return capture

def main():
    global capture
    id_ = [0, 1, 2]
    t1 = threading.Thread(target=get_fram, args=())
    t2 = threading.Thread(target=detect_face, args=(id_[0],))
    t3 = threading.Thread(target=detect_face, args=(id_[1],))
    t4 = threading.Thread(target=detect_face, args=(id_[2],))
    t5 = threading.Thread(target=landmark_face, args=(id_[0],))
    t6 = threading.Thread(target=landmark_face, args=(id_[1],))
    t7 = threading.Thread(target=landmark_face, args=(id_[2],))
    t8 = threading.Thread(target=show, args=())

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    # t6.start()


if __name__ == "__main__":
    global_flag = True
    anchors = get_anchors([640, 640], [[0.025, 0.05], [0.1, 0.2], [0.4, 0.8]],
                          [8, 16, 32])
    fram_queue = Queue()
    origin_queue = Queue()
    randmark_queue = Queue()
    capture = camera_set()
    main()
