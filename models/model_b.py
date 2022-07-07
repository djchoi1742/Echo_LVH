import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Activation, Conv2D, Conv3D, Dense, AvgPool2D
from tensorflow.keras.layers import MaxPooling2D, Bidirectional, ConvLSTM2D, MaxPooling3D
from tensorflow.keras.models import *
from tensorflow import keras
import sys, re
import tensorflow.keras.backend as backend
import sklearn.metrics

sys.path.append('/workspace/bitbucket/Echocard')


def design_scope(class_name):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', 'Classifier', model_scope)
    return model_scope, classifier_scope


def conv3d_base(x, f_num, k_size, stride, is_training=True, padding='same', af='relu', init='he_normal'):
    out = Conv3D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    out = BatchNormalization(trainable=is_training)(out)
    return out


def conv2d_base(x, f_num, k_size, stride, is_training=True, padding='same', af='relu', init='he_normal'):
    out = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    out = BatchNormalization(trainable=is_training)(out)
    return out


def conv2d_time(x, f_num, k_size, stride, is_training=True, padding='same', af='relu', init='he_normal'):
    out = TimeDistributed(Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init))(x)
    out = BatchNormalization(trainable=is_training)(out)
    return out


def conv2d_lstm(x, f_num, k_size, stride, is_training=True, padding='same', af='relu', init='he_normal'):
    out = ConvLSTM2D(filters=f_num, kernel_size=k_size, strides=stride, padding=padding, return_sequences=True)(x)
    out = BatchNormalization(trainable=is_training)(out)
    return out


class Model09:  # 2D CNN
    def __init__(self, input_size, f_num, is_training, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 3

        self.label_num = 3
        self.img_h, self.img_w, self.img_c = input_size
        block_num = len(f_num)

        out = self.images
        for i in range(block_num):
            out = conv2d_base(x=out, f_num=f_num[i], k_size=3, stride=1, is_training=is_training)
            out = conv2d_base(x=out, f_num=f_num[i], k_size=3, stride=1, is_training=is_training)
            out = MaxPooling2D(pool_size=(2, 2))(out)

        conv_out = conv2d_base(x=out, f_num=f_num[-1], k_size=3, stride=1, is_training=is_training)
        print('conv_out: ', conv_out)

        bot_gap = tf.reduce_mean(conv_out, axis=[1, 2], keepdims=False)
        logits = layers.Dense(self.label_num, activation=None)(bot_gap)
        probs = tf.nn.sigmoid(logits)

        self.cam_layer_name = 'batch_normalization_10'  # time_distributed_14 or bidirectional
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.get_layer('dense').output, self.model.output])


class Model12:  # 2D CNN-LSTM
    def __init__(self, input_size, f_num, is_training, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        self.label_num = 3
        self.img_z, self.img_h, self.img_w, self.img_c = input_size
        block_num = len(f_num)

        out = self.images
        for i in range(block_num):
            out = conv2d_time(x=out, f_num=f_num[i], k_size=3, stride=1, is_training=is_training)
            out = conv2d_time(x=out, f_num=f_num[i], k_size=3, stride=1, is_training=is_training)
            out = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(out)

        conv_lstm = Bidirectional(ConvLSTM2D(filters=f_num[-1], kernel_size=3, strides=1, padding='same',
                                  return_sequences=False), merge_mode='ave')(out)
        print('conv_lstm: ', conv_lstm)

        bot_gap = tf.reduce_mean(conv_lstm, axis=[1, 2], keepdims=False)
        logits = layers.Dense(self.label_num, activation=None)(bot_gap)
        probs = tf.nn.sigmoid(logits)

        self.cam_layer_name = 'time_distributed_14'  # time_distributed_14 or bidirectional
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.get_layer('dense').output, self.model.output, logits])

        self.frozen_model = keras.Model(inputs=self.images, outputs=bot_gap)


class Model16:  # 3D CNN
    def __init__(self, input_size, f_num, is_training, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        self.label_num = 3
        self.img_z, self.img_h, self.img_w, self.img_c = input_size
        block_num = len(f_num)

        # p_size = [2, 2, 2, 2, 1]  # for 16 slices  [8, 4, 2, 1, 1]
        p_size = [2, 2, 3, 1, 1]  # for 12 slices  [6, 3, 1, 1, 1]

        out = self.images
        print('**', out)
        for i in range(block_num):
            out = conv3d_base(x=out, f_num=f_num[i], k_size=3, stride=1, is_training=is_training)
            out = conv3d_base(x=out, f_num=f_num[i], k_size=3, stride=1, is_training=is_training)
            out = MaxPooling3D(pool_size=(p_size[i], 2, 2))(out)
            print('**', i, out)

        conv_out = conv3d_base(x=out, f_num=f_num[-1], k_size=3, stride=1, is_training=is_training)
        bot_gap = tf.reduce_mean(conv_out, axis=[1, 2, 3], keepdims=False)
        logits = layers.Dense(self.label_num, activation=None)(bot_gap)

        probs = tf.nn.sigmoid(logits)

        self.cam_layer_name = 'max_pooling3d_3'
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.get_layer('dense').output, self.model.output,
                                              logits])

        self.frozen_model = keras.Model(inputs=self.images, outputs=bot_gap)



def focal_loss_sigmoid(y_true, y_pred, alpha=0.5, gamma=0.):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    smooth = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, smooth, 1.0 - smooth)

    fcl_loss = -y_true*(1-alpha)*((1-y_pred)**gamma)*tf.math.log(y_pred) - \
               (1-y_true)*alpha*(y_pred**gamma)*tf.math.log(1-y_pred)
    lbl_loss = tf.reduce_mean(fcl_loss, axis=0)

    return lbl_loss


def calculate_auc(y, x):
    try:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y, x, drop_intermediate=False)
        auc_value = sklearn.metrics.auc(fpr, tpr)
    except:
        auc_value = 0.0000
    return auc_value


if __name__ == '__main__':
    import os, logging
    import numpy as np
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.disable(logging.WARNING)

    # gpu = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation

    infer = Model12(input_size=[12, 256, 256, 1], f_num=[64, 96, 128, 192, 256], is_training=True)
    # infer = Model09(input_size=[256, 256, 1], f_num=[64, 96, 128, 192, 256], is_training=True)
    # infer = Model16(input_size=[16, 256, 256, 1], f_num=[64, 96, 128, 192, 256], is_training=True)
    # infer = Model16(input_size=[12, 256, 256, 1], f_num=[64, 96, 128, 192, 256],
    #                 p_size=[2, 2, 3, 1, 1], is_training=True)
    # infer.model.summary()


    if False:
        model_weights = infer.model.get_weights()
        idx = 0
        for weight in model_weights:
            print(idx, weight.shape)
            idx += 1
