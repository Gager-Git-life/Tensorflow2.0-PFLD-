import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Model.convs import _Conv2D, _inverted_residual_block, CBAMblock, SEblock


def PFLD(input_shape=(112, 112, 3)):

    # (-1, 112, 112, 3)
    inputs = layers.Input(shape=input_shape)
    # (-1, 112, 112, 3) --> (-1, 56, 56, 64)
    conv1 = _Conv2D(inputs=inputs, filters=64, kernel_size=3, strides=2, activation='hs')
    # (-1, 56, 56, 64) --> (-1, 56, 56, 64)
    # d_conv2 = _Conv2D(inputs=conv1, filters=64, kernel_size=3, strides=1, activation='hs')
    # (-1, 56, 56, 64) --> (-1, 28, 28, 64)
    conv3 = _inverted_residual_block(inputs=conv1, in_filter=64, out_filter=64, kernel_size=(3,3), stride=2,
                                   expand_rate=2, repeat=3, use_se=True, activation='')
    featu = conv3
    # (-1, 28, 28, 64) --> (-1, 14, 14, 128)
    conv4 = _inverted_residual_block(inputs=conv3, in_filter=64, out_filter=128, kernel_size=(3,3), stride=2,
                                   expand_rate=2, repeat=1, use_se=True)

    # (-1, 14, 14, 128) --> (-1, 14, 14, 128)
    conv5 = _inverted_residual_block(inputs=conv4, in_filter=128, out_filter=128, kernel_size=(3, 3), stride=1,
                                   expand_rate=4, repeat=3, use_se=True)

    # (-1, 14, 14, 128) --> (-1, 14, 14, 16)
    conv6 = _inverted_residual_block(inputs=conv5, in_filter=128, out_filter=16, kernel_size=(3,3), stride=1,
                                   expand_rate=2, repeat=1, use_se=True)

    # (-1, 14, 14, 16) --> (-1, 7, 7, 32)
    conv7 = _Conv2D(inputs=conv6, filters=32, kernel_size=(3,3), strides=2, activation='hs')
    # (-1, 7, 7, 32) --> (-1, 1, 1, 128)
    conv8 = _Conv2D(inputs=conv7, filters=128, kernel_size=(7,7), strides=1, activation='hs', padding='valid')

    # (-1, 14, 14, 16) --> (-1, 1, 1, 16) --> (-1, 16)
    s1 = layers.AveragePooling2D(pool_size=(14, 14))(conv6)
    s1 = layers.Flatten()(s1)
    # (-1, 7, 7, 32) --> (-1, 1, 1, 32) --> (-1, 32)
    s2 = layers.AveragePooling2D(pool_size=(7,7))(conv7)
    s2 = layers.Flatten()(s2)
    # (-1, 1, 1, 128) --> (-1, 128)
    s3 = layers.Flatten()(conv8)

    # (-1, 1, 1, 176)
    concat = tf.concat([s1, s2, s3], axis=-1)
    # print(concat.shape)
    landmark = layers.Dense(196)(concat)
    # landmark = layers.BatchNormalization()(landmark)
    landmark = tf.nn.tanh(landmark)


    model = keras.Model(inputs=inputs, outputs=[featu, landmark])
    return model




def AuxiliaryNet(input_shape=(28, 28, 64)):

    # (-1, 28, 28, 64)
    inputs = layers.Input(shape=input_shape)

    # (-1, 28, 28, 64) --> (-1, 14, 14, 128)
    conv1 = _Conv2D(inputs=inputs, filters=128, kernel_size=3, strides=2)
    # conv1 = SEblock(conv1)
    # (-1, 14, 14, 128) --> (-1, 14, 14, 128)
    conv2 = _Conv2D(inputs=conv1, filters=128, kernel_size=3, strides=1)
    # conv2 = SEblock(conv2)
    # (-1, 14, 14, 128) --> (-1,  7,  7,  32)
    conv3 = _Conv2D(inputs=conv2, filters=32, kernel_size=3, strides=2)
    # conv3 = SEblock(conv3)
    # (-1,  7,  7,  32) --> (-1, 1, 1, 128)
    conv4 = _Conv2D(inputs=conv3, filters=128, kernel_size=7, strides=1, padding='valid')

    conv5 = layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')(conv4)

    conv5 = layers.Flatten()(conv5)

    out1 = layers.Dense(32)(conv5)
    out = layers.Dense(3)(out1)
    # out = tf.reshape(out, shape=(-1, out.shape[-1]))

    model = keras.Model(inputs=inputs, outputs=out)

    return model

