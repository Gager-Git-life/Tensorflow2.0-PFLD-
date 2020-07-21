import tensorflow as tf
from Model.utils import Normalization
import numpy as np

def pfld_loss(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):

    landmarks = Normalization(landmarks)

    weight_angle = tf.reduce_sum(tf.map_fn(lambda x: tf.nn.sigmoid(abs(x))- 0.5, euler_angle_gt - angle), axis=1)

    attributes_w_n = tf.cast(attribute_gt[:, 1:6], dtype=tf.float32)
    mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    mat_ratio = tf.map_fn(lambda x: (tf.cond(x > 0, lambda: 1 / x, lambda: float(train_batchsize))), mat_ratio)
    attributes_w_n = tf.convert_to_tensor(attributes_w_n * mat_ratio)
    weight_attribute = tf.reduce_sum(attributes_w_n, axis=1)

    l2_distant = tf.reduce_sum(tf.square((landmark_gt - landmarks))*10, axis=1)
    return tf.reduce_mean(weight_angle * weight_attribute + l2_distant), tf.reduce_mean(l2_distant), tf.reduce_mean(weight_angle)

def pfld_loss_(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):

    weight_angle = tf.reduce_sum(tf.map_fn(lambda x: tf.sin(x), euler_angle_gt - angle), axis=1)

    # attributes_w_n = tf.cast(attribute_gt[:, 1:6], dtype=tf.float32)
    # mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    # mat_ratio = tf.map_fn(lambda x: (tf.cond(x > 0, lambda: 1 / x, lambda: float(train_batchsize))), mat_ratio)
    # attributes_w_n = tf.convert_to_tensor(attributes_w_n * mat_ratio)
    # weight_attribute = tf.reduce_sum(attributes_w_n, axis=1)

    l2_distant = tf.reduce_sum(tf.square((landmark_gt - landmarks)*112), axis=1)
    return tf.reduce_mean(l2_distant), tf.reduce_mean(l2_distant), tf.reduce_mean(weight_angle)