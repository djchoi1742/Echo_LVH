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
# ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')

ROOT_PATH = '/workspace/Echocard/'
RAW_PATH = os.path.join(ROOT_PATH, 'RAW')
INFO_PATH = os.path.join(ROOT_PATH, 'info', 'dataset')
sys.path.append('/workspace/bitbucket/Echocard')


def select_train_groups(x, group_list):
    out = True if x in group_list else False
    return out


def extract_group(df, data_type):
    group_values = df['GROUP'].tolist()
    type_groups = [*map(int, re.split(',', data_type))]  
    df_subset = df[[*map(lambda x: select_train_groups(x, type_groups), group_values)]]  
    return df_subset


def view_label(df, data_type, raw_path, view, shuffle=False):
    v = str(view)
    group_values = df['GROUP'].tolist()
    type_groups = [*map(int, re.split(',', data_type))]
    df = df[[*map(lambda x: select_train_groups(x, type_groups), group_values)]]

    df = df[['INDEX', 'GROUP', 'FOLDER_NAME', 'LABEL1', 'LABEL2', 'LABEL3', 'EQUAL_Z',
             'VIEW'+v, 'PIXEL_X'+v, 'PIXEL_Y'+v, 'PIXEL_Z'+v, 'SX'+v, 'SY'+v, 'SZ'+v, 'TRUNC'+v]]

    if shuffle:
        df = df.iloc[np.random.RandomState(seed=20210306).permutation(len(df))].reset_index(drop=True)

    df_names = ['P%04d' % x for x in df['INDEX'].tolist()]

    view_paths = df.apply(lambda row: os.path.join(raw_path, row['FOLDER_NAME'], row['VIEW' + v]), axis=1)
    v_paths = view_paths.values.tolist()

    slices, equal_zs, frame_times = df['PIXEL_Z'+v].astype(int).tolist(), df['EQUAL_Z'].tolist(), df['SZ'+v].tolist()
    truncs = df['TRUNC'+v].astype(int).tolist()
    labels = np.array(df[['LABEL1', 'LABEL2', 'LABEL3']])

    return v_paths, labels, df_names, slices, frame_times, equal_zs, truncs


def view_3d(views, f_slices, f_times, zs, num_slices, is_trunc):
    def extract_slice(f_time, z, f_slice, view):
        if z == 0:
            ft_vector = np.repeat(f_time, f_slice).tolist()

            ft_vector[0] = 0  # first slice

        elif z == 1:
            ft_vector = dcm.read_file(view)['0018', '1065'].value
            ft_vector = [float(x) for x in ft_vector][0:f_slice] if is_trunc else [float(x) for x in ft_vector]
        else:
            raise ValueError

        ft_sum = np.cumsum(ft_vector).tolist()
        total_ft = ft_sum[-1]

        select_ft = np.repeat(total_ft / (num_slices - 1), num_slices)
        select_ft[0] = 0
        cum_select_ft = np.cumsum(select_ft).tolist()

        f_time_slices = [*map(lambda x: select_slice(x, ft_sum), cum_select_ft)]

        return f_time_slices

    extract_slices = [*map(extract_slice, f_times, zs, f_slices, views)]

    return extract_slices


def select_slice(x, y):
    return np.argmin(np.abs(x - np.array(y)))


def view_extract(views, labels, names, f_slices, f_times, zs, num_slices=12):
    def extract_path(paths, total_slices):
        paths_list = np.repeat(paths, total_slices)
        return paths_list.tolist()

    def extract_slice(f_time, z, f_slice, view):
        if z == 0:
            ft_vector = np.repeat(f_time, f_slice).tolist()
            ft_vector[0] = 0  # first slice
        elif z == 1:
            ft_vector = dcm.read_file(view)['0018', '1065'].value
            ft_vector = [float(x) for x in ft_vector]
        else:
            raise ValueError

        ft_sum = np.cumsum(ft_vector).tolist()
        total_ft = ft_sum[-1]

        select_ft = np.repeat(total_ft / (num_slices - 1), num_slices)
        select_ft[0] = 0
        cum_select_ft = np.cumsum(select_ft).tolist()

        f_time_slices = [*map(lambda x: select_slice(x, ft_sum), cum_select_ft)]

        return f_time_slices

    def expand_label(label):
        return np.reshape(np.repeat(label, num_slices), (len(labels)*num_slices, -1), order='F')

    def expand_name(x):
        return np.repeat(x, num_slices).tolist()

    view_dcms = [*map(lambda x: extract_path(x, num_slices), views)]
    extract_slices = [*map(extract_slice, f_times, zs, f_slices, views)]
    name_expands = [*map(expand_name, names)]

    view_dcms = list(itertools.chain.from_iterable(view_dcms))
    extract_slices = list(itertools.chain.from_iterable(extract_slices))
    name_expands = list(itertools.chain.from_iterable(name_expands))
    name_slices = ['_'.join([i, '%03d' % (j + 1)]) for i, j in zip(name_expands, extract_slices)]

    label_slices = np.squeeze(np.apply_along_axis(expand_label, 0, labels))

    return view_dcms, extract_slices, label_slices, name_slices


def pixel_array_truncated(ds, idx, x0, x1, y0, y1):  # only SNUBH data
    pixel_bytes = bytearray()
    for frame in decode_data_sequence(ds.PixelData):
        try:
            im = Image.open(io.BytesIO(frame))
            if 'YBR' in ds.PhotometricInterpretation:
                im.draft('YCbCr', (ds.Rows, ds.Columns))
            pixel_bytes.extend(im.tobytes())
        except:
            continue
    arr = np.frombuffer(pixel_bytes, pixel_dtype(ds))
    arr = np.reshape(arr, (-1, ds.Rows, ds.Columns, ds.SamplesPerPixel))
    preprocess_arr = arr[idx, y0:y1, x0:x1, :]
    return preprocess_arr


def image3d_preprocess(view, idx, lbl, name, trunc, img_h, img_w, img_c, shift=False, bin_lbl=False):
    ds = dcm.read_file(view.numpy().decode())
    seq_ds = ds['0018', '6011'][0]
    spatial_format = seq_ds['0018', '6012'].value

    if spatial_format == 0:
        seq_ds = ds['0018', '6011'][1]

    min_x0 = seq_ds['0018', '6018'].value  # Region Location Min X0
    min_y0 = seq_ds['0018', '601a'].value  # Region Location Min Y0
    max_x1 = seq_ds['0018', '601c'].value  # Region Location Max X1
    max_y1 = seq_ds['0018', '601e'].value  # Region Location Max Y1

    ph_d_x = abs(seq_ds['0018', '602c'].value)  # Physical Delta X (spacing)
    ph_d_y = seq_ds['0018', '602e'].value  # Physical Delta Y (spacing)

    if not trunc:
        valid_img = ds.pixel_array[idx.numpy(), min_y0:max_y1, min_x0:max_x1, :]
    else:
        valid_img = pixel_array_truncated(ds, idx.numpy(), min_x0, max_x1, min_y0, max_y1)

    z, y, x = valid_img.shape[0:3]
    cx, cy = int(x / 2), int(y / 2)

    x1, y1 = int(cx - 6 / ph_d_x), int(cy - 6 / ph_d_y)
    x2, y2 = int(cx + 6 / ph_d_x), int(cy + 6 / ph_d_y)

    if shift:
        shift_x = np.random.randint(x // 40)
        shift_y = np.random.randint(y // 40)

        # shifting direction
        shift_x = -shift_x if np.random.rand() <= 0.5 else shift_x
        shift_y = -shift_y if np.random.rand() <= 0.5 else shift_y

        x1 = x1 - shift_x
        y1 = y1 - shift_y
        x2 = x2 - shift_x
        y2 = y2 - shift_y

    img = valid_img[:, max(0, y1):min(y, y2), max(0, x1):min(x, x2)]
    img = skimage.transform.resize(img, [z, img_h, img_w, img_c], preserve_range=True)

    if bin_lbl:
        lbl = np.sum(lbl)

    lbl = tf.cast(lbl, tf.int32)

    return img, lbl, name


def image3d_dataset(img, lbl, name, img_z, img_c, augment=False, c_min=1., c_max=2., b_val=.4):
    img.set_shape([img_z, None, None, img_c])

    ori_image = tf.unstack(img, axis=0)

    def img_preprocess(each_seq):
        each_seq = tf.image.per_image_standardization(each_seq)
        if augment:
            each_seq = tf.image.random_contrast(each_seq, c_min, c_max)
            each_seq = tf.image.random_brightness(each_seq, b_val)
        return each_seq

    img = tf.stack([*map(img_preprocess, ori_image)], axis=0)

    return img, lbl, name


def py_image3d_preprocess(view, idx, lbl, name, trunc, img_h, img_w, img_c, shift=False, bin_label=False):
    imgs, lbls, names = tf.py_function(image3d_preprocess,
                                       [view, idx, lbl, name, trunc, img_h, img_w, img_c, shift, bin_label],
                                       [tf.float32, tf.int32, tf.string])
    return imgs, lbls, names


def view3d_tf_read(view, index, label, name, trunc, img_h, img_w, img_z, img_c, shuffle, shift, augment, bin_label):

    dataset = tf.data.Dataset.from_tensor_slices((view, index, label, name, trunc))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=8*10, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x, y, z, w, v: py_image3d_preprocess(x, y, z, w, v, img_h, img_w, img_c,
                                                                      shift, bin_label))
    dataset = dataset.map(lambda x, y, z: image3d_dataset(x, y, z, img_z, img_c, augment))
    return dataset


class DataSettingV1:
    def __init__(self, df, train_name, val_name, view, train=False, img_z=12,
                 img_h=256, img_w=256, img_c=1, bin_label=False, **kwargs):
        val_view, val_labels, val_names, val_slices, val_times, val_zs, val_tcs = \
            view_label(df, val_name, RAW_PATH, view, False)
        val_slices = view_3d(val_view, val_slices, val_times, val_zs, img_z, val_tcs)

        self.val = view3d_tf_read(view=val_view, index=val_slices, label=val_labels, name=val_names,
                                  trunc=val_tcs, img_h=img_h, img_w=img_w, img_z=img_z, img_c=img_c,
                                  shuffle=False, shift=False, augment=False, bin_label=bin_label)

        if train:
            train_view, train_labels, train_names, train_slices, train_times, train_zs, train_tcs = \
                view_label(df, train_name, RAW_PATH, view, True)

            train_slices = view_3d(train_view, train_slices, train_times, train_zs, img_z, train_tcs)

            self.train = view3d_tf_read(view=train_view, index=train_slices, label=train_labels, name=train_names,
                                        trunc=train_tcs, img_h=img_h, img_w=img_w, img_z=img_z, img_c=img_c,
                                        shuffle=True, shift=True, augment=True, bin_label=bin_label)


def view_image(dataset):
    view_path = os.path.join(ROOT_PATH, config.exp_name, 'view', 'view'+str(config.view))
    if not os.path.exists(view_path): os.makedirs(view_path)
    view_data_path = os.path.join(view_path, config.val_name)
    if not os.path.exists(view_data_path): os.makedirs(view_data_path)

    num_examples = dataset.val.cardinality().numpy()
    print('num examples: ', num_examples)

    for batch in dataset.val.batch(config.batch_size):
        img, lbl, name = batch

        for j in range(len(name.numpy())):
            lbl1, lbl2, lbl3 = lbl[j].numpy()

            if np.sum(lbl1 + lbl2 + lbl3) == 0:
                name_color = 'darkslategray'
            if lbl1 == 1:
                name_color = 'royalblue'
            if lbl2 == 1:
                name_color = 'seagreen'
            if lbl3 == 1:
                name_color = 'darkgoldenrod'

            fig, ax = plt.subplots(3, 4, figsize=(4 * 2, 3 * 2))
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            z_axis = 3 if config.trans else 1
            for k in range(img.shape[z_axis]):
                r_idx, c_idx = k // 4, k % 4

                img_slice = img.numpy()[j, :, :, k, :] if config.trans else img.numpy()[j, k, :, :, :]
                ax[r_idx, c_idx].imshow(np.squeeze(img_slice), cmap='gray')
                ax[r_idx, c_idx].set_title('_'.join([name[j].numpy().decode(), '%02d' % k,
                                                     str(lbl1), str(lbl2), str(lbl3)]), color=name_color, fontsize=10)

            png_name = '_'.join([name[j].numpy().decode()]) + '.png'

            fig_name = os.path.join(view_data_path, png_name)
            plt.savefig(fig_name, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('data setting')
    setup_config.add_argument('--excel_name', type=str, dest='excel_name', default='Echo04', help='excel_name')
    setup_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp004', help='exp index')
    setup_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,5')
    setup_config.add_argument('--val_name', type=str, dest='val_name', default='4')
    setup_config.add_argument('--batch_size', type=int, dest='batch_size', default=4, help='batch size')
    setup_config.add_argument('--view', type=int, dest='view', default=1)
    setup_config.add_argument('--trans', type=lambda x: x.title() in str(True), dest='trans', default=False)
    setup_config.add_argument('--image_height', type=int, dest='image_height', default=256)
    setup_config.add_argument('--image_width', type=int, dest='image_width', default=256)
    setup_config.add_argument('--image_depth', type=int, dest='image_depth', default=12)
    setup_config.add_argument('--image_channel', type=int, dest='image_channel', default=1)
    setup_config.add_argument('--bin_label', type=lambda x: x.title() in str(True), dest='bin_label', default=False)

    config, unparsed = parser.parse_known_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    df_path = os.path.join(INFO_PATH, config.excel_name + '.xlsx')
    dfs = pd.read_excel(df_path)
    dfs = dfs[dfs['INVALID'] == 0]

    d_set = DataSettingV1(df=dfs, train_name=config.train_name, val_name=config.val_name, view=config.view,
                          img_h=config.image_height, img_w=config.image_width, img_c=config.image_channel,
                          img_z=config.image_depth, train=True, bin_label=config.bin_label)

    view_image(d_set)

    if False:
        for batch in d_set.train.batch(2):
            img, lbl, name = batch
            print(img.shape, lbl.numpy(), name.numpy())
