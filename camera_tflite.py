import cv2
import time
import os, sys
import numpy as np
import tensorflow as tf
import threading
from multiprocessing import Process
from multiprocessing import Queue
from scipy.special import softmax, expit
from Model.utils import color_
from Model.infer_gpu import get_anchors, nms_oneclass, decode_bbox, decode_landm

fram_queue = Queue()
out_queue = Queue()
global_flag = True

capture = cv2.VideoCapture(-1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def model_predict(interpreter, input, output, img):

    # print(input, '\n', img.shape)
    interpreter.set_tensor(input[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output[0]['index'])

    return output

def get_model():
    interpreter_pfld = tf.lite.Interpreter(model_path='models/tflite/pfld_infer.tflite')
    interpreter_retinaface = tf.lite.Interpreter(model_path='models/tflite/retinaface.tflite')

    interpreter_pfld.allocate_tensors()
    interpreter_retinaface.allocate_tensors()

    pfld_input_details = interpreter_pfld.get_input_details()
    pfld_output_details = interpreter_pfld.get_output_details()

    retina_input_details = interpreter_retinaface.get_input_details()
    retina_output_details = interpreter_retinaface.get_output_details()

    print(retina_input_details, '\n', retina_output_details)
    print(pfld_input_details, '\n', pfld_output_details)
    return (interpreter_pfld, pfld_input_details, pfld_output_details), \
           (interpreter_retinaface, retina_input_details, retina_output_details)


def detect_face(retina_models, pfld_models,
                anchors, index,
                obj_thresh=0.7,
                nms_threshold=0.4,
                variances=[0.1, 0.2]):

    global fram_queue, out_queue, global_flag
    while(global_flag):
        if(fram_queue.qsize() >= 1):
            print('-----')
            draw_img = fram_queue.get()

            interpreter_pfld, pfld_input, pfld_output = pfld_models
            interpreter_retina, retina_input, retina_output = retina_models

            draw_img = cv2.copyMakeBorder(draw_img, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=0)
            """ resize """
            img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
            """ normlize """
            det_img = ((img / 255. - 0.5) / 1)[None, ...]
            """ infer """
            det_img = det_img.astype(np.float32)
            predictions = model_predict(interpreter_retina, retina_input, retina_output, det_img)
            print(predictions.shape)
            """ parser """
            bbox, landm, clses = np.split(predictions[0], [4, -2], 1)
            """ softmax class"""
            clses = softmax(clses, -1)
            score = clses[:, 1]
            # print('score.shape:{}'.format(score.shape))
            """ decode """
            bbox = decode_bbox(bbox, anchors, variances)
            bbox = bbox * np.tile([640, 640], [2])
            # print('bbox.shape:{}'.format(bbox.shape))
            """ filter low score """
            inds = np.where(score > obj_thresh)[0]
            # print('inds.shape:{}'.format(inds.shape))
            bbox = bbox[inds]
            score = score[inds]
            # print('score.shape:{}'.format(score.shape))
            # print('bbox.shape:{}'.format(bbox.shape))
            """ keep top-k before NMS """
            order = np.argsort(score)[::-1]
            bbox = bbox[order]
            score = score[order]
            # print('score.shape:{}'.format(score.shape))
            # print('bbox.shape:{}'.format(bbox.shape))
            """ do nms """
            keep = nms_oneclass(bbox, score, nms_threshold)
            # print('score.shape:{}'.format(score.shape))
            # print('bbox.shape:{}'.format(bbox.shape))

            for b, s in zip(bbox[keep].astype(int), score[keep]):
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
                if croped_wh[0] == croped_wh[1] and min(croped_wh) > 10:
                    croped_img = cv2.resize(croped_img, (112, 112))
                    croped_img = ((croped_img / 255. - 0.5) / 1)[None, ...]
                    croped_img = croped_img.astype(np.float32)
                    landmarks = model_predict(interpreter_pfld, pfld_input, pfld_output, croped_img)

                    s_point = np.array([cx - halfw, cy - halfw])
                    # print((cx,cy), s_point, croped_wh)

                    for i, landm in enumerate(np.reshape(expit(landmarks), (-1, 2)) * croped_wh):
                        color = color_(i)
                        cv2.circle(draw_img, tuple((s_point + landm).astype(int)), 1, color)
            out_queue.put(draw_img)
        else:
            # print(fram_queue.qsize())
            pass


def get_fram():
    global fram_queue, capture, global_flag
    while(capture.isOpened() and global_flag):
        ret, img = capture.read()
        if(ret):
            print('输入一帧：{}'.format(fram_queue.qsize()))
            fram_queue.put(img)

def show():
    global out_queue
    global capture
    global global_flag
    old_time = time.time()
    while(True):
        if(out_queue.qsize() >= 1):
            img = out_queue.get()

            cv2.putText(img, "FPS {0}".format(str(1 / (time.time() - old_time))), (10, 10), 2, 0.5,
                        (255, 0, 255),
                        1)
            old_time = time.time()
            cv2.imshow('fram', img)
            if cv2.waitKey(1) == ord('q'):
                global_flag = False
                break

    capture.release()
    cv2.destroyAllWindows()
    # sys.exit(0)

if __name__ == "__main__":

    anchors = get_anchors([640, 640], [[0.025, 0.05], [0.1, 0.2], [0.4, 0.8]],
                          [8, 16, 32])



    pfld_models, retina_models = get_model()

    # count = 0
    # start_time = time.time()
    # pre_image = None
    #
    # while (capture.isOpened()):
    #     ret, img = capture.read()
    #     draw_img = detect_face(retina_models, pfld_models, anchors, img)
    #
    #     pre_image = draw_img
    #     count += 1
    #     cv2.putText(pre_image, "FPS {0}".format(str(count / (time.time() - start_time))), (10, 10), 2, 0.5,
    #                 (255, 0, 255),
    #                 1)
    #     if (time.time() - start_time >= 1):
    #         start_time = time.time()
    #         count = 0
    #
    #     cv2.imshow('frame', draw_img)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # capture.release()
    # cv2.destroyAllWindows()


    p0 = threading.Thread(target=get_fram, args=())
    p5 = threading.Thread(target=show, args=())
    p1 = Process(target=detect_face, args=(retina_models, pfld_models, anchors, 0))
    p2 = Process(target=detect_face, args=(retina_models, pfld_models, anchors, 1))
    p3 = Process(target=detect_face, args=(retina_models, pfld_models, anchors, 2))
    p4 = Process(target=detect_face, args=(retina_models, pfld_models, anchors, 3))

    p0.setDaemon(daemonic=True)
    p5.setDaemon(daemonic=True)
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()


    # t6.start()

    print('ok')

