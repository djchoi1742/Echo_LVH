import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import sys
import pydicom as dcm
import skimage.transform, scipy.misc
import re
import warnings
import argparse
import matplotlib.pyplot as plt
import itertools
import io
from PIL import Image
from pydicom.encaps import decode_data_sequence
from pydicom.pixel_data_handlers.util import pixel_dtype

warnings.filterwarnings('ignore')

ROOT_PATH = '/workspace/Echocard/'
RAW_PATH = os.path.join(ROOT_PATH, 'RAW')
INFO_PATH = os.path.join(ROOT_PATH, 'info', 'dataset')
sys.path.append('/workspace/bitbucket/Echocard')


def load_frozen_npy(exp_name, model_name, serial, data_name):
    npy_path = os.path.join(ROOT_PATH, exp_name, model_name, 'result-%03d' % int(serial), data_name+'.npy')
    load_npy = np.load(npy_path, allow_pickle=True).item()
    x, y, name = load_npy.values()  # order: x, y, name
    return x, y, name


def merge_to_dataset(x, y, name, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y, name))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=8*10, reshuffle_each_iteration=True)

    return dataset


class DataSettingV2:
    def __init__(self, exp_name, model_name, serial, train_name, val_name, train):
        serial = re.split(',', serial)
        serial1, serial2, serial3, serial4, serial5 = serial

        val_x1, val_y, val_n = load_frozen_npy(exp_name, model_name, serial1, val_name)
        val_x2, _, _ = load_frozen_npy(exp_name, model_name, serial2, val_name)
        val_x3, _, _ = load_frozen_npy(exp_name, model_name, serial3, val_name)
        val_x4, _, _ = load_frozen_npy(exp_name, model_name, serial4, val_name)
        val_x5, _, _ = load_frozen_npy(exp_name, model_name, serial5, val_name)

        val_x = np.concatenate([val_x1, val_x2, val_x3, val_x4, val_x5], axis=1)
        self.val = merge_to_dataset(val_x, val_y, val_n, shuffle=False)

        if train:
            train_x1, train_y, train_n = load_frozen_npy(exp_name, model_name, serial1, train_name)
            train_x2, _, _ = load_frozen_npy(exp_name, model_name, serial2, train_name)
            train_x3, _, _ = load_frozen_npy(exp_name, model_name, serial3, train_name)
            train_x4, _, _ = load_frozen_npy(exp_name, model_name, serial4, train_name)
            train_x5, _, _ = load_frozen_npy(exp_name, model_name, serial5, train_name)
            train_x = np.concatenate([train_x1, train_x2, train_x3, train_x4, train_x5], axis=1)
            self.train = merge_to_dataset(train_x, train_y, train_n, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('data setting')
    setup_config.add_argument('--excel_name', type=str, dest='excel_name', default='Echo07', help='excel_name')
    setup_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp007', help='exp index')
    setup_config.add_argument('--model_name', type=str, dest='model_name', default='Model12', help='model name')
    setup_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,5')
    setup_config.add_argument('--val_name', type=str, dest='val_name', default='4')
    setup_config.add_argument('--batch_size', type=int, dest='batch_size', default=2, help='batch size')
    setup_config.add_argument('--view', type=int, dest='view', default=1)

    config, unparsed = parser.parse_known_args()
    d_set = DataSettingV2(exp_name=config.exp_name, model_name=config.model_name, serial='11,12,13,14,15',
                          train_name=config.train_name, val_name=config.val_name, train=True)
