# Neural network to aggregate the outputs of 5 echocardiograms
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


class Model15:
    def __init__(self, input_size, block_num, is_training, **kwargs):
        self.features = layers.Input(input_size)
        self.label_num = 3

        f_num = self.features.shape[-1]

        out = self.features
        for i in range(block_num):
            out = layers.Dense(f_num, activation=tf.nn.relu)(out)
            out = BatchNormalization(trainable=is_training)(out)

        logits = layers.Dense(self.label_num, activation=None)(out)
        probs = tf.nn.sigmoid(logits)

        self.model = Model(inputs=self.features, outputs=probs)


def focal_loss_sigmoid(y_true, y_pred, alpha=0.5, gamma=0.):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    smooth = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, smooth, 1.0)

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

    infer = Model15(input_size=[256*5], block_num=2, is_training=True)
    infer.model.summary()
