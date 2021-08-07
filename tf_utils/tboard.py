import os
import tensorflow as tf


def tensorboard_create(log_path):
    train_log_path = os.path.join(log_path, 'train')
    if not os.path.exists(train_log_path): os.makedirs(train_log_path)

    val_log_path = os.path.join(log_path, 'val')
    if not os.path.exists(val_log_path): os.makedirs(val_log_path)

    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    val_summary_writer = tf.summary.create_file_writer(val_log_path)

    return train_summary_writer, val_summary_writer


def board_record_value(summary, values, epoch):
    with summary.as_default():
        for key in values.keys():
            tf.summary.scalar(key, values[key], step=epoch)
