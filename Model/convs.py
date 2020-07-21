from Model.activatinos import *

def _Conv2D(inputs, filters, kernel_size, strides, activation='relu', bn='LN', padding='same', deepthwise=False):
    if(deepthwise):
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    else:
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    # x = layers.BatchNormalization()(x)

    if(isinstance(bn, str)):
        if(bn == 'IN'):
            x = InstanceNormalization()(x)
        elif(bn == 'LN'):
            x = LayerInsNorm()(x)
        else:
            x = layers.BatchNormalization()(x)

    if(activation == 'relu'):
        return layers.ReLU()(x)
    elif(activation == 'hs'):
        return h_swish(x)
    elif(activation == ''):
        return tf.keras.activations.linear(x)
    else:
        return x

def SEblock(inputs, alpha=1):
    input_channel = inputs.shape[-1]
    branch = layers.GlobalAvgPool2D()(inputs)
    branch = layers.Dense(units=input_channel*alpha)(branch)
    branch = layers.Activation(tf.nn.relu)(branch)
    branch = layers.Dense(units=input_channel)(branch)
    branch = h_sigmoid(branch)
    branch = tf.expand_dims(branch, axis=1)
    branch = tf.expand_dims(branch, axis=1)

    output = inputs * branch
    return output

def _Bottleneck(inputs, in_filter, out_filter, kernel_size, stride, expand_ratio, use_se, use_res, activation=''):

    up_channel = in_filter * expand_ratio
    x = _Conv2D(inputs=inputs, filters=up_channel, kernel_size=(1, 1), strides=(1, 1))

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', depth_multiplier=1)(x)
    # x = layers.BatchNormalization()(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # 添加se模块
    if(use_se):
        x = SEblock(x)

    # x = layers.Conv2D(filters=out_filter, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    # x = layers.Activation(tf.keras.activations.linear)(x)
    x = _Conv2D(inputs=x, filters=out_filter, kernel_size=(1,1), strides=(1,1), activation=activation)

    if use_res and in_filter == out_filter:
        x = layers.Add()([inputs, x])

    return x

def _inverted_residual_block(inputs, in_filter, out_filter, kernel_size, stride, expand_rate, repeat, use_se, activation=''):

  x = _Bottleneck(inputs, in_filter, out_filter, kernel_size, stride, expand_rate, use_se, False, activation)
  for _ in range(1, repeat):
    x = _Bottleneck(x, out_filter, out_filter, kernel_size, 1, expand_rate, use_se, True, '')

  return x


def CAblock(inputs, ratio=8):

    input_channel = inputs.shape[-1]

    maxpool = layers.GlobalMaxPool2D()(inputs)
    maxpool = tf.reshape(maxpool, shape=(-1, 1, 1, maxpool.shape[-1]))
    avgpool = layers.GlobalAveragePooling2D()(inputs)
    avgpool = tf.reshape(avgpool, shape=(-1, 1, 1, avgpool.shape[-1]))

    shared_fc1 = layers.Conv2D(filters=int(input_channel/8), kernel_size=1, strides=1)
    shared_fc2 = layers.Conv2D(filters=input_channel, kernel_size=1, strides=1)
    shared_act = layers.Activation(tf.nn.relu)

    max_out = shared_fc2(shared_act(shared_fc1(maxpool)))
    avg_out = shared_fc2(shared_act(shared_fc1(avgpool)))

    outputs = h_swish(max_out + avg_out)

    return outputs

def SAblock(inputs, kernel_size=7):

    maxpool = tf.reduce_mean(inputs, axis=-1)
    avgpool = tf.reduce_max(inputs, axis=-1)

    concat = tf.stack([maxpool, avgpool], axis=-1)

    outputs = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same', strides=1)(concat)
    outputs = h_swish(outputs)

    return outputs


def CBAMblock(inputs):

    outputs = SAblock(inputs) * inputs
    outputs = CAblock(outputs) * outputs

    return outputs