import os
import sys
import cv2
import copy
import time
import glob
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list', type=str,default='data/train_data/list.txt')
    parser.add_argument('--test_list', type=str, default='data/test_data/list.txt')
    parser.add_argument('--seed',type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='models1/model_test')
    parser.add_argument('--learning_rate', type=float, default=0.00014)
    parser.add_argument('--lr_epoch', type=str, default='10,20,30,40,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example',action='store_false')
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--first_train', type=bool, default=False)
    parser.add_argument('--gen_mark_png', type=str, default="./save_landmark_png")
    parser.add_argument('--checkpoint_dir', type=str, default="./training_checkpoints")
    parser.add_argument('--logs', type=str, default="./logs")
    return parser.parse_args(argv)

def Normalization(landmarks):
    landmarks = tf.reshape(landmarks, shape=(-1, 98, 2))
    xy_max = tf.expand_dims(tf.reduce_max(landmarks, axis=1), axis=1)
    xy_min = tf.expand_dims(tf.reduce_min(landmarks, axis=1), axis=1)
    # print(xy_min.shape)
    landmarks = (landmarks - xy_min) / (xy_max - xy_min)
    landmarks = tf.reshape(landmarks, shape=(-1, 196))
    # print(tf.reduce_max(landmarks, axis=1), tf.reduce_min(landmarks, axis=1))
    return landmarks

def color_(num):
    if(num <= 32):
        return (0, 0, 255)
    elif(num <= 50):
        return (0 ,250,154)
    elif(num <= 54):
        return (0,0,0)
    elif(num <= 59):
        return (255,255,0)
    elif(num <= 75):
        return (255, 20, 147)
    elif(num <= 87):
        return (0, 255, 0)
    elif(num <= 95):
        return (255, 0, 0)
    elif(num == 96 or num == 97):
        return (255, 20, 147)

def Draw_lamdmark(image, landmark, color=(255, 0, 0)):
    # print(image.shape)
    # print(np.max(landmark), np.min(landmark))
    landmark = landmark.reshape(-1, 2)
    _img = copy.deepcopy(image)
    for i, (x, y) in enumerate(landmark.astype(np.int32)):
        color = color_(i)
        cv2.circle(_img, (x, y), 1, color)
    return _img


def generate_and_save_images(model, epoch, test_inputs, test_landmarks):
    _, landmarks = model(test_inputs, training=False)
    landmarks = Normalization(landmarks)
    landmarks = np.array(landmarks*112)
    test_landmarks = np.array(test_landmarks*112)
    test_inputs = (np.array(test_inputs) + 1.0)/2
    inputs = copy.deepcopy(test_inputs)

    plt.figure(figsize=(9,9))
    for i in range(len(landmarks)):
        plt.subplot(4, 2, i*2+1)
        plt.axis('off')
        plt.imshow(Draw_lamdmark(inputs[i], landmarks[i]))
        plt.subplot(4, 2, i*2+2)
        plt.axis('off')
        plt.imshow(Draw_lamdmark(test_inputs[i], test_landmarks[i], color=(0,255,0)))

    plt.savefig('save_landmark_png/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close()

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(len(gpus))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print('[INFO]>>> current use cpu!!!')

class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, self.fps))

        sys.stdout.flush()

def prepare(args):
    if(not os.path.exists(args.logs)):
        print("[INFO]>>> create {}".format(args.logs))
    else:
        os.system("rm -rf {}/*".format(args.logs))
    if(not os.path.exists(args.gen_mark_png)):
        print("[INFO]>>> create {}".format(args.gen_mark_png))
    else:
        os.system("rm -rf {}/*".format(args.gen_mark_png))
    if(not os.path.exists(args.checkpoint_dir)):
        print("[INFO]>>> create {}".format(args.checkpoint_dir))
    else:
        os.system("rm -rf {}/*".format(args.checkpoint_dir))

def train_init(args):
    prepare(args)

def get_gif(pic_dir, pic_format, save_path='dcgan.gif'):

    with imageio.get_writer(save_path, mode='I') as writer:
      filenames = glob.glob('{}/{}'.format(pic_dir, pic_format))
      filenames = sorted(filenames)
      last = -1
      for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)