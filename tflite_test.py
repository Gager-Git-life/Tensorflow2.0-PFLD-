import os
import sys
import cv2
import time
import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Model.datasets import DateSet
from scipy.special import softmax, expit
from Model.utils import parse_arguments, Normalization, color_

args = parse_arguments(sys.argv[1:])
test_dataset, num_test_file = DateSet(args.test_list, args, False)
batch_test_dataset = test_dataset.batch(60)
test_iterator = iter(batch_test_dataset)
# test_data  = next(test_iterator)
# test_images = test_data[0]
# test_landmarks = test_data[1]
# print(test_images.shape, test_images.dtype)

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  print(interpreter.get_input_details())
  interpreter.set_tensor(tensor_index, image)
  # input_tensor = interpreter.set_tensor(tensor_index)()[0]
  # input_tensor = tf.expand_dims(input_tensor, axis=0)
  # print(input_tensor.shape)
  # for i in range(image.shape[0]):
  # input_tensor = image
  # print(input_tensor.shape)

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    return  output

def Draw_lamdmark(image, landmark, flag=True):
    # print(image.shape)
    landmark = landmark.reshape(-1, 2)
    img = copy.deepcopy(image)
    if(flag):
        for i, (x, y) in enumerate(landmark.astype(np.int32)):
            color = color_(i)
            cv2.circle(img, (x, y), 1, color)
        return img
    else:
        return image

def generate_and_save_images(test_inputs, test_landmarks, landmarks, step):
    # landmarks = Normalization(landmarks)
    landmarks = np.array(landmarks*112)
    # landmarks = np.array(landmarks)

    test_inputs = np.array(test_inputs)
    test_landmarks = np.array(test_landmarks * 112)
    inputs = copy.deepcopy(test_inputs)

    plt.figure(figsize=(10,10))
    for i in range(len(test_inputs)):
        plt.subplot(60, 3, i*3+1)
        plt.axis('off')
        plt.imshow(Draw_lamdmark(test_inputs[i], landmarks[i], flag=False))
        plt.subplot(60, 3, i*3+2)
        plt.axis('off')
        plt.imshow(Draw_lamdmark(inputs[i], landmarks[i]))
        plt.subplot(60, 3, i*3+3)
        plt.axis('off')
        plt.imshow(Draw_lamdmark(test_inputs[i], test_landmarks[i]))

    plt.savefig('pic/tilite_test_{}.png'.format(step))
    plt.close()

interpreter = tf.lite.Interpreter(model_path='models/tflite/pfld_infer.tflite')
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())
# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作为输入测试 TensorFlow Lite 模型。
for i in range(1):
    test_images, test_landmarks, _, _ = next(test_iterator)
    input_data = test_images
    bt = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 函数 `get_tensor()` 会返回一份张量的拷贝。
    # 使用 `tensor()` 获取指向张量的指针。
    landmarks = interpreter.get_tensor(output_details[0]['index'])
    landmarks = expit(landmarks)
    print(np.max(landmarks), np.min(landmarks))
    print('[INFO]>>> 用时:{}'.format(time.time() - bt))
    generate_and_save_images(test_images, test_landmarks, landmarks, i)

