import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
import trade_data

def create_inputs(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64*8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size*64,
                                  min_after_dequeue=cfg.batch_size*32, allow_smaller_final_batch=False)

    return (x, y)

def load_mnist(path, is_training):
    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)
    teX = tf.convert_to_tensor(teX / 255., tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY

def load_trade(is_training):
    if is_training:
        data = trade_data.read_data_sets(cfg.dataset,one_hot=False)
        return data.train
    else:
        data = trade_data.read_test_sets(cfg.test_dataset, one_hot=False)
        return data

def get_shuffle_batch_data(is_training):
    data= load_trade(is_training)
    trX = data.images
    trY = data.labels
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=cfg.num_threads,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)
    print(X.shape,"--",Y.shape)
    return(X, Y,data.num_examples)

def get_batch_data(is_training):
    data = load_trade(is_training)
    trX=data.images
    trY = data.labels
    data_queues = tf.train.slice_input_producer([trX, trY],shuffle=False)
    thread_num = 1
    if is_training:
        thread_num=cfg.num_threads

    X, Y = tf.train.batch(data_queues, num_threads=thread_num,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  allow_smaller_final_batch=True)
    print(X.shape,"--",Y.shape)
    return(X, Y,data.num_examples)

def get_pred_data():
    trX,dt= trade_data.get_csv_data2(cfg.test_dataset)
    data_queues = tf.train.slice_input_producer([trX],shuffle=False)
    X= tf.train.batch(data_queues, num_threads=1,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  allow_smaller_final_batch=True)
    datanum = trX.shape[0]
    print("get_pred_data:",X.shape)
    return(X,dt,datanum)