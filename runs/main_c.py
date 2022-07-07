# Training and validation of neural network
import os, sys, logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import argparse, json
import datetime

import skimage.transform
import re

sys.path.append('/workspace/bitbucket/Echocard')

parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/Echocard')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='Echo07')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp007')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,5')  # ex: 'train'
main_config.add_argument('--val_name', type=str, dest='val_name', default='4')  # ex: 'val'
main_config.add_argument('--pre_model', type=str, dest='pre_model', default='Model12')
main_config.add_argument('--pre_serial', type=str, dest='pre_serial', default='11,12,13,14,15')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model15')
main_config.add_argument('--block_num', type=int, dest='block_num', default=2)
main_config.add_argument('--serial', type=int, dest='serial', default=1)
main_config.add_argument('--max_keep', type=int, dest='max_keep', default=3)  # only use training
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.00005)
main_config.add_argument('--decay_steps', type=int, dest='decay_steps', default=400)  # prev: 200
main_config.add_argument('--decay_rate', type=int, dest='decay_rate', default=0.94)
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=20)
main_config.add_argument('--epoch', type=int, dest='epoch', default=10)
main_config.add_argument('--alpha', type=float, dest='alpha', default=0.05)
main_config.add_argument('--gamma', type=float, dest='gamma', default=2.)
main_config.add_argument('--is_png', type=lambda x: x.title() in str(True), dest='is_png', default=False)


config, unparsed = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation

serial_str = '%03d' % config.serial
log_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'logs-%s' % serial_str)
result_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%s' % serial_str)
plot_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'plot-%s' % serial_str)

if not os.path.exists(log_path): os.makedirs(log_path)
if not os.path.exists(result_path): os.makedirs(result_path)
if not os.path.exists(plot_path): os.makedirs(plot_path)


ROOT_PATH = '/workspace/Echocard/'
RAW_PATH = os.path.join(ROOT_PATH, 'RAW')
INFO_PATH = os.path.join(ROOT_PATH, 'info', 'dataset')


from data.setup_c import DataSettingV2
import models.model_c as model_ref
import tf_utils.tboard as tboard

d_set = DataSettingV2(exp_name=config.exp_name, model_name=config.pre_model, serial=config.pre_serial,
                      train_name=config.train_name, val_name=config.val_name, train=config.train)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)


if config.train:
    train_db = d_set.train.batch(config.batch_size)
    train_length = d_set.train.cardinality().numpy()
    print('train length: ', train_length)


infer_name = config.model_name
infer = getattr(model_ref, infer_name)(input_size=[1280], block_num=config.block_num, is_training=config.train)

model = infer.model

loss_fn = model_ref.focal_loss_sigmoid
loss_weight = np.array([1/3, 1/3, 1/3])


def training():
    info_log = {
        'EXCEL_FILE': config.excel_name,
        'MODEL_NAME': config.model_name,
        'SERIAL': config.serial,
        'PRE_SERIAL': config.pre_serial,
        'PRE_MODEL': config.pre_model,
        'TRAIN_NAME': config.train_name,
        'VAL_NAME': config.val_name,
        'BLOCK_NUM': config.block_num,
        'BATCH_SIZE': config.batch_size,
        'LEARNING_RATE': config.learning_rate,
        'DECAY_STEPS': config.decay_steps,
        'DECAY_RATE': config.decay_rate,
        'EPOCH': config.epoch
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)

    result_name = '_'.join([config.exp_name, config.model_name, serial_str])+'.csv'
    auc_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'AUC_MEAN': pd.Series(),
                            'AUC1': pd.Series(), 'AUC2': pd.Series(), 'AUC3': pd.Series()})

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=True)

    optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

    perf_per_epoch, max_perf_per_epoch, max_current_step = [], [], []
    log_string = ''
    start_time = datetime.datetime.now()

    try:
        for epoch in range(1, config.epoch+1):
            train_loss, train_l1_loss, train_l2_loss, train_l3_loss = [], [], [], []
            train_x1, train_x2, train_x3, train_y1, train_y2, train_y3 = [], [], [], [], [], []
            for train_step, (x, lbl, name) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    probs = model(x)
                    lbl = tf.cast(lbl, tf.float32)

                    train_loss_batch = loss_fn(lbl, probs)
                    total_loss_batch = loss_weight * train_loss_batch
                    lbl1_loss, lbl2_loss, lbl3_loss = total_loss_batch.numpy()

                grads = tape.gradient(train_loss_batch, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(total_loss_batch)
                train_l1_loss.append(lbl1_loss)
                train_l2_loss.append(lbl2_loss)
                train_l3_loss.append(lbl3_loss)

                prob1, prob2, prob3 = tf.unstack(probs, axis=1)
                lbl1, lbl2, lbl3 = tf.unstack(lbl, axis=1)

                train_x1.extend(prob1.numpy())
                train_y1.extend(lbl1.numpy())
                train_x2.extend(prob2.numpy())
                train_y2.extend(lbl2.numpy())
                train_x3.extend(prob3.numpy())
                train_y3.extend(lbl3.numpy())

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} L1: {2:.4f} L2: {3:.4f} L3: {4:.4f} ({5})\r'.
                                 format(train_step, np.sum(total_loss_batch), lbl1_loss, lbl2_loss, lbl3_loss, epoch))

            train_loss_mean = np.mean(train_loss)
            train_l1_loss_mean = np.mean(train_l1_loss)
            train_l2_loss_mean = np.mean(train_l2_loss)
            train_l3_loss_mean = np.mean(train_l3_loss)

            train_auc1 = model_ref.calculate_auc(train_y1, train_x1)
            train_auc2 = model_ref.calculate_auc(train_y2, train_x2)
            train_auc3 = model_ref.calculate_auc(train_y3, train_x3)

            train_record = {'Loss': train_loss_mean,
                            'Loss1': train_l1_loss_mean, 'Loss2': train_l2_loss_mean, 'Loss3': train_l3_loss_mean,
                            'AUC1': train_auc1, 'AUC2': train_auc2, 'AUC3': train_auc3}

            val_loss, val_l1_loss, val_l2_loss, val_l3_loss = [], [], [], []
            val_x1, val_x2, val_x3, val_y1, val_y2, val_y3 = [], [], [], [], [], []
            for val_step, (x, lbl, name) in enumerate(val_db):
                sys.stdout.write('Evaluation [{0}/{1}]\r'.format(val_step, val_length // config.batch_size))

                val_probs = model(x)
                lbl = tf.cast(lbl, tf.float32)

                val_loss_batch = loss_fn(lbl, val_probs)
                val_total_loss_batch = loss_weight * val_loss_batch
                lbl1_loss, lbl2_loss, lbl3_loss = val_total_loss_batch.numpy()

                val_loss.append(val_total_loss_batch)
                val_l1_loss.append(lbl1_loss)
                val_l2_loss.append(lbl2_loss)
                val_l3_loss.append(lbl3_loss)

                val_prob1, val_prob2, val_prob3 = tf.unstack(val_probs, axis=1)
                val_lbl1, val_lbl2, val_lbl3 = tf.unstack(lbl, axis=1)

                val_x1.extend(val_prob1.numpy())
                val_y1.extend(val_lbl1.numpy())
                val_x2.extend(val_prob2.numpy())
                val_y2.extend(val_lbl2.numpy())
                val_x3.extend(val_prob3.numpy())
                val_y3.extend(val_lbl3.numpy())

            val_loss_mean = np.mean(val_loss)
            val_l1_loss_mean = np.mean(val_l1_loss)
            val_l2_loss_mean = np.mean(val_l2_loss)
            val_l3_loss_mean = np.mean(val_l3_loss)

            val_auc1 = model_ref.calculate_auc(val_y1, val_x1)
            val_auc2 = model_ref.calculate_auc(val_y2, val_x2)
            val_auc3 = model_ref.calculate_auc(val_y3, val_x3)

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += 'Time Elapsed:{0}'.format(time_elapsed.split('.')[0])

            val_record = {'Loss': val_loss_mean,
                          'Loss1': val_l1_loss_mean, 'Loss2': val_l2_loss_mean, 'Loss3': val_l3_loss_mean,
                          'AUC1': val_auc1, 'AUC2': val_auc2, 'AUC3': val_auc3}

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            print('Epoch:%s Train-Loss:%.4f L1:%.4f L2:%.4f L3:%.4f AUC1:%.4f AUC2:%.4f AUC3:%.4f'
                  ' Val-Loss:%.4f L1:%.4f L2:%.4f L3:%.4f AUC1:%.4f AUC2:%.4f AUC3:%.4f ' %
                  (epoch, train_loss_mean, train_l1_loss_mean, train_l2_loss_mean, train_l3_loss_mean,
                   train_auc1, train_auc2, train_auc3,
                   val_loss_mean, val_l1_loss_mean, val_l2_loss_mean, val_l3_loss_mean,
                   val_auc1, val_auc2, val_auc3) + log_string)

            log_string = ''
            val_auc = np.mean([val_auc1, val_auc2, val_auc3])
            perf_per_epoch.append(val_auc)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_auc)
                model.save(weight_path)

                auc_csv.loc[epoch] = weight_path, val_auc, val_auc1, val_auc2, val_auc3

            elif val_auc > min(auc_csv['AUC_MEAN'].tolist()):
                os.remove(auc_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                auc_csv = auc_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_auc)

                model.save(weight_path)

                auc_csv.loc[epoch] = weight_path, val_auc, val_auc1, val_auc2, val_auc3
            auc_csv.to_csv(os.path.join(result_path, result_name))

            if epoch == config.epoch: break

    except KeyboardInterrupt:
        print('Result saved')
        auc_csv.to_csv(os.path.join(result_path, result_name))


def validation():
    png_path = os.path.join(plot_path, config.val_name)
    if not os.path.exists(png_path): os.makedirs(png_path)

    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial])+'.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('AUC_MEAN', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    probs = np.zeros([len(all_ckpt_paths), val_length, infer.label_num])
    names = []

    ckpt_idx = 0
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)
        print('ckpt: ', ckpt)
        val_loss, val_l1_loss, val_l2_loss, val_l3_loss = [], [], [], []
        val_x1, val_x2, val_x3, val_y1, val_y2, val_y3 = [], [], [], [], [], []

        for step, (x, lbl, name) in enumerate(val_db):
            sys.stdout.write('Evaluation [{0}/{1}]\r'.format(step, val_length // config.batch_size))

            with tf.GradientTape() as tape:
                val_probs = model(x)

            lbl = tf.cast(lbl, tf.float32)

            val_loss_batch = loss_fn(lbl, val_probs)
            val_total_loss_batch = loss_weight * val_loss_batch
            lbl1_loss, lbl2_loss, lbl3_loss = val_total_loss_batch.numpy()

            val_prob1, val_prob2, val_prob3 = tf.unstack(val_probs, axis=1)
            label1, label2, label3 = tf.unstack(lbl, axis=1)

            val_x1.extend(val_prob1.numpy())
            val_y1.extend(label1.numpy())

            val_x2.extend(val_prob2.numpy())
            val_y2.extend(label2.numpy())

            val_x3.extend(val_prob3.numpy())
            val_y3.extend(label3.numpy())

            val_loss.append(val_loss_batch)
            val_l1_loss.append(lbl1_loss)
            val_l2_loss.append(lbl2_loss)
            val_l3_loss.append(lbl3_loss)

            probs[ckpt_idx, step * config.batch_size:step * config.batch_size + len(lbl)] = val_probs

            if ckpt_idx == 0:
                names.extend(name.numpy().astype('str'))

        ckpt_idx += 1

    val_auc1 = model_ref.calculate_auc(val_y1, val_x1)
    val_auc2 = model_ref.calculate_auc(val_y2, val_x2)
    val_auc3 = model_ref.calculate_auc(val_y3, val_x3)

    print('Validation AUC1: %.3f AUC2: %.3f AUC3: %.3f' % (val_auc1, val_auc2, val_auc3))

    result_csv = pd.DataFrame({'NUMBER': names, 'LABEL1': val_y1, 'PROB1': val_x1,
                               'LABEL2': val_y2, 'PROB2': val_x2, 'LABEL3': val_y3, 'PROB3': val_x3})

    result_name = '_'.join([config.model_name, config.val_name, serial_str, '%03d' % config.num_weight])+'.csv'
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()

