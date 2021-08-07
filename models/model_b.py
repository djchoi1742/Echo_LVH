import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Activation, Conv2D, Conv3D, Dense, AvgPool2D
from tensorflow.keras.layers import MaxPooling2D, Bidirectional, ConvLSTM2D, GlobalAveragePooling2D
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


def bottleneck_layer_time(input_x, growth_k, is_training):
    out = TimeDistributed(BatchNormalization(trainable=is_training))(input_x)
    out = TimeDistributed(Activation('relu'))(out)
    out = TimeDistributed(Conv2D(filters=4*growth_k, kernel_size=1, strides=1, padding='same', activation=None,
                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                          kernel_initializer=tf.keras.initializers.he_normal()))(out)
    out = TimeDistributed(BatchNormalization(trainable=is_training))(out)
    out = TimeDistributed(Activation('relu'))(out)
    out = TimeDistributed(Conv2D(filters=growth_k, kernel_size=3, strides=1, padding='same', activation=None,
                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                          kernel_initializer=tf.keras.initializers.he_normal()))(out)
    return out


def se_block_time(input_x, reduction_ratio=16):
    squeeze = tf.reduce_mean(input_x, axis=[2, 3], keepdims=True)  # global average pooling
    excitation = TimeDistributed(Dense(units=squeeze.shape[-1] // reduction_ratio,
                                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 activation='relu'))(squeeze)
    excitation = TimeDistributed(Dense(units=squeeze.shape[-1],
                                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 activation='sigmoid'))(excitation)
    return excitation


def dense_block(input_x, layer_name, rep, growth_k, is_training, use_se=False, r_ratio=16):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer_time(input_x, growth_k, is_training)
        layers_concat.append(x)

        for i in range(rep - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer_time(x, growth_k, is_training)
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=3)

        if use_se:
            excitation = se_block_time(x, r_ratio)
            x = x * excitation
    return x


def transition_layer_time(input_x, layer_name, is_training, theta=0.5, reduction_ratio=16, last_layer=False):
    with tf.name_scope(layer_name):
        in_channel = input_x.shape[-1]
        out = TimeDistributed(BatchNormalization(trainable=is_training))(input_x)
        out = TimeDistributed(Activation('relu'))(out)
        out = TimeDistributed(Conv2D(filters=int(in_channel*theta), kernel_size=1, strides=1,
                              padding='same', activation=None,
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal()))(out)
        if last_layer is False:
            excitation = se_block_time(out, reduction_ratio)
            se_out = out * excitation
            avg_pool = TimeDistributed(AvgPool2D(pool_size=(2, 2), strides=2, padding='same'))(se_out)
            print(avg_pool)
        else:
            avg_pool = out

    return avg_pool


def bottleneck_layer_3d(input_x, growth_k, is_training, use_dropout=False, dropout_rate=0.2):
    out = BatchNormalization(trainable=is_training)(input_x)
    out = Activation('relu')(out)
    out = Conv3D(filters=4*growth_k, kernel_size=(1, 1, 1), strides=1, padding='same', activation=None,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                 kernel_initializer=tf.keras.initializers.he_normal())(out)
    out = BatchNormalization(trainable=is_training)(out)
    out = Activation('relu')(out)
    out = Conv3D(filters=growth_k, kernel_size=(4, 4, 2), strides=1, padding='same', activation=None,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                 kernel_initializer=tf.keras.initializers.he_normal())(out)
    if use_dropout:
        out = layers.Dropout(rate=dropout_rate, is_training=is_training)(out)

    return out


def dense_block_3d(input_x, layer_name, rep, growth_k, is_training, use_dropout=False,
                   use_se=False, reduction_ratio=16):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer_3d(input_x, growth_k, is_training, use_dropout)
        layers_concat.append(x)

        for i in range(rep - 1):
            x = tf.concat(layers_concat, axis=4)
            x = bottleneck_layer_3d(x, growth_k, is_training, use_dropout)
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=4)

        if use_se:
            excitation = se_block_3d(x, reduction_ratio)
            x = x * excitation
    return x


def se_block_3d(input_x, reduction_ratio=16):
    squeeze = tf.reduce_mean(input_x, axis=[1, 2, 3], keepdims=True)  # global average pooling
    excitation = layers.Dense(units=squeeze.shape[-1] // reduction_ratio,
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='relu')(squeeze)
    excitation = layers.Dense(units=squeeze.shape[-1],
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='sigmoid')(excitation)
    return excitation


def transition_layer_3d(input_x, layer_name, is_training, theta=0.5, reduction_ratio=16, last_layer=False):
    with tf.name_scope(layer_name):
        in_channel = input_x.shape[-1]
        out = layers.BatchNormalization(trainable=is_training)(input_x)
        out = layers.Activation('relu')(out)
        out = layers.Conv3D(filters=int(in_channel*theta), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                            padding='same', activation=None,
                            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                            kernel_initializer=tf.keras.initializers.he_normal())(out)

        if last_layer is False:
            excitation = se_block_3d(out, reduction_ratio)
            se_out = out * excitation
            avg_pool = layers.AvgPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(se_out)
            # print(avg_pool)
        else:
            avg_pool = out

    return avg_pool


class Model10:
    def __init__(self, input_size, is_training=False, growth_k=32, theta=0.5,
                 block_rep='2,2,2,2,2', use_se=False, **kwargs):
        # super(InferenceModel01, self).__init__()
        self.model_scope, _ = design_scope(class_name=type(self).__name__)

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)

        self.is_training = is_training
        self.label_num = 3

        self.img_h, self.img_w, self.img_d, self.img_c = input_size

        self.images = layers.Input((self.img_h, self.img_w, self.img_d, self.img_c))

        first_conv = layers.Conv3D(filters=2*growth_k, kernel_size=(4, 4, 2), strides=(1, 1, 1),
                                   padding='same', activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   )(self.images)
        first_batch = layers.BatchNormalization(trainable=self.is_training)(first_conv)
        first_relu = layers.Activation('relu', name='conv1/relu')(first_batch)
        first_pool = layers.MaxPooling3D(pool_size=(4, 4, 2), strides=(2, 2, 2),
                                         padding='same')(first_relu)

        dsb = first_pool
        for i in range(0, block_num-1):
            dsb = dense_block_3d(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                                 growth_k=growth_k, use_se=use_se, is_training=self.is_training)
            print('%d DB: ' % (i+1), dsb)
            dsb = transition_layer_3d(input_x=dsb, layer_name='Transition'+str(i+1),
                                      theta=theta, is_training=self.is_training)
            print('%d Trans: ' % (i+1), dsb)

        self.last_dsb = dense_block_3d(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                                       rep=block_rep_list[-1], growth_k=growth_k,
                                       use_se=use_se, is_training=self.is_training)

        last_bn = layers.BatchNormalization(trainable=self.is_training)(self.last_dsb)
        self.bn_relu = layers.Activation('relu', name='last_bn_relu')(last_bn)

        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2, 3], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)
        flatten = tf.reduce_mean(self.last_pool, axis=[1, 2, 3], keepdims=False)
        self.fc = layers.Dense(units=flatten.shape[-1], activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                               kernel_initializer=tf.keras.initializers.he_normal()
                               )(flatten)  # activation='relu'

        self.logits = layers.Dense(units=self.label_num, activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal()
                                   )(self.fc)
        self.prob = layers.Activation('sigmoid')(self.logits)
        self.model = keras.Model(inputs=self.images, outputs=self.prob)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer('last_bn_relu').output, self.model.output])


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


class Model08:
    def __init__(self, input_size, f_num, is_training, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        self.label_num = 4
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
        probs = tf.nn.softmax(logits)

        self.cam_layer_name = 'time_distributed_14'  # time_distributed_14 or bidirectional
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.get_layer('dense').output, self.model.output])



class Model09:
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

        print('conv_lstm: ', conv_out)

        bot_gap = tf.reduce_mean(conv_out, axis=[1, 2], keepdims=False)
        logits = layers.Dense(self.label_num, activation=None)(bot_gap)
        probs = tf.nn.sigmoid(logits)

        self.cam_layer_name = 'batch_normalization_10'  # time_distributed_14 or bidirectional
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.get_layer('dense').output, self.model.output])


class Model11:
    def __init__(self, input_size, f_num, is_training, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        self.label_num = 3
        self.img_z, self.img_h, self.img_w, self.img_c = input_size

        conv_1 = conv2d_time(x=self.images, f_num=f_num[0], k_size=3, stride=1, is_training=is_training)
        conv_1 = conv2d_time(x=conv_1, f_num=f_num[0], k_size=3, stride=1, is_training=is_training)
        down_1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv_1)

        conv_2 = conv2d_time(x=down_1, f_num=f_num[1], k_size=3, stride=1, is_training=is_training)
        conv_2 = conv2d_time(x=conv_2, f_num=f_num[1], k_size=3, stride=1, is_training=is_training)
        down_2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv_2)

        conv_3 = conv2d_time(x=down_2, f_num=f_num[2], k_size=3, stride=1, is_training=is_training)
        conv_3 = conv2d_time(x=conv_3, f_num=f_num[2], k_size=3, stride=1, is_training=is_training)
        down_3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv_3)

        conv_4 = conv2d_time(x=down_3, f_num=f_num[3], k_size=3, stride=1, is_training=is_training)
        conv_4 = conv2d_time(x=conv_4, f_num=f_num[3], k_size=3, stride=1, is_training=is_training)
        down_4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv_4)

        # conv_lstm = ConvLSTM2D(filters=f_num[3], kernel_size=3, strides=1, padding='same',
        #                        return_sequences=False)(down_4)

        conv_lstm = Bidirectional(ConvLSTM2D(filters=f_num[3], kernel_size=3, strides=1, padding='same',
                                  return_sequences=False), merge_mode='ave')(down_4)

        bot_gap = tf.reduce_mean(conv_lstm, axis=[1, 2], keepdims=False)

        logits = layers.Dense(self.label_num, activation=None)(bot_gap)
        probs = tf.nn.sigmoid(logits)

        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer('time_distributed_11').output, self.model.output])


class Model12:
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
                                              self.model.get_layer('dense').output, self.model.output,
                                              logits])

        self.frozen_model = keras.Model(inputs=self.images, outputs=bot_gap)
        import pdb; pdb.set_trace()


class Model13:
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

        conv_cls = Conv2D(filters=self.label_num, kernel_size=1, strides=1, padding='same', activation=None,
                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                          kernel_initializer=tf.keras.initializers.he_normal())(conv_lstm)

        logits = tf.reduce_mean(conv_cls, axis=[1, 2], keepdims=False)
        probs = tf.nn.sigmoid(logits)

        self.cam_layer_name = 'conv2d_10'
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              logits, self.model.output])


class Model14:
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

        conv_cls = Conv2D(filters=self.label_num, kernel_size=1, strides=1, padding='same', activation=None,
                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                          kernel_initializer=tf.keras.initializers.he_normal())(conv_lstm)

        logits = tf.reduce_logsumexp(conv_cls, axis=[1, 2], keepdims=False)
        probs = tf.nn.sigmoid(logits)

        self.cam_layer_name = 'conv2d_10'
        self.model = Model(inputs=self.images, outputs=probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              logits, self.model.output])


def focal_loss_sigmoid(y_true, y_pred, alpha=0.5, gamma=0.):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
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

    # infer = Model12(input_size=[12, 256, 256, 1], f_num=[64, 96, 128, 192, 256], is_training=True)
    # infer = Model09(input_size=[256, 256, 1], f_num=[64, 96, 128, 192, 256], is_training=True)
    infer = Model12(input_size=[12, 256, 256, 1], f_num=[64, 96, 128, 192, 256], is_training=True)
    infer.model.summary()

    if False:
        model_weights = infer.model.get_weights()
        idx = 0
        for weight in model_weights:
            print(idx, weight.shape)
            idx += 1