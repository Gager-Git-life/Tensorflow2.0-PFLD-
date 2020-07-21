import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

"""
https://blog.csdn.net/u013289254/article/details/99690730
"""

weight_initializer = tf.random_normal_initializer(0., 0.02)
weight_regularizer = keras.regularizers.l2(0.0002)

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


def h_swish(x):
    return x * h_sigmoid(x)


#BN
def BN(input, bn_training, name):
    def bn(input, bn_training, name):
        output = layers.BatchNormalization(input, training=bn_training, name=name)
        return output
    return tf.cond(
        bn_training,
        lambda: bn(input, bn_training=True, name=name),
        lambda: bn(input, bn_training=False, name=name),
    )

#LN
def Layernorm(x, gamma, beta):
    results = 0.
    eps = 1e-5
    x = tf.transpose(x, [0, 3, 1, 2])  # [B,C,H,W]
    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    results = tf.transpose(results, [0, 2, 3, 1])
    return results

#IN
def Instancenorm(x, gamma, beta):
    # x_shape:[B, H, W, C]
    results = 0.
    eps = 1e-5
    x = tf.transpose(x, [0, 3, 1, 2]) #[B,C,H,W]
    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    #
    results = tf.transpose(results, [0, 2, 3, 1])
    return results
#GN
def GN(x, traing_flag, scope, G=16, esp=1e-5):
    # tranpose:[bs,h,w,c]to[bs,c,h,w]follwing the paper
    x = tf.transpose(x, [0, 3, 1, 2])
    N, C, H, W = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gama and beta
    gama = tf.Variable(scope + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
    beta = tf.Variable(scope  + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
    gama = tf.reshape(gama, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])
    output = tf.reshape(x, [-1, C, H, W]) * gama + beta
    ## tranpose:[bs,c,h,w]to[bs,h,w,c]follwing the paper
    output = tf.transpose(output, [0, 2, 3, 1])
    return output

#SN
def SwitchableNorm(x, gamma, beta, w_mean, w_var):
    # x_shape:[B, H, W, C]
    results = 0.
    eps = 1e-5
    x = tf.transpose(x, [0, 3, 1, 2])  # [B,C,H,W]
    mean_in = np.mean(x, axis=(2, 3), keepdims=True)
    var_in = np.var(x, axis=(2, 3), keepdims=True)

    mean_ln = np.mean(x, axis=(1, 2, 3), keepdims=True)
    var_ln = np.var(x, axis=(1, 2, 3), keepdims=True)

    mean_bn = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var_bn = np.var(x, axis=(0, 2, 3), keepdims=True)

    mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
    var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn

    x_normalized = (x - mean) / np.sqrt(var + eps)
    results = gamma * x_normalized + beta
    results = tf.transpose(results, [0, 2, 3, 1])
    return results

class InstanceNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name=None):
        super(InstanceNormalization, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=weight_initializer,
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset



class LayerInsNorm(keras.layers.Layer):
    def __init__(self, name=None):
        super(LayerInsNorm, self).__init__(name=name)
        self.epsilon = 1e-5

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(0.0),
            trainable=True,
            constraint=keras.constraints.MinMaxNorm(0.0, 1.0))
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(1.0),
            trainable=True)
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(1.0),
            trainable=True)

    def call(self, x):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))
        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.epsilon))
        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        return x_hat * self.gamma + self.beta