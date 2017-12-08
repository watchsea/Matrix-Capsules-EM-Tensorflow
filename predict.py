"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
from config import cfg
import numpy as np
import math
import utils
import trade_data

import capsnet_em as net

def main(_):
    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    with tf.Graph().as_default():
        batch_x,dt,datanum = utils.get_pred_data()
        num_batches_test = math.ceil(datanum / cfg.batch_size)
        print("total data:",datanum, ", run count:", num_batches_test,",  dt:",dt)
        #print(batch_x)

        output = net.build_arch(batch_x, coord_add, is_train=False)
        predict = tf.argmax(output, axis=1)

        saver = tf.train.Saver()

        sess = tf.Session()
        tf.train.start_queue_runners(sess=sess)
        ckpt = tf.train.get_checkpoint_state(cfg.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)

        for i in range(num_batches_test):
            y_pred,output1 = sess.run([predict,output])

            if i% 10 ==0:
                print("step:",i,"/",num_batches_test) #,",",np.shape(y_pred),np.shape(output1))
            if i==0:
                y_pred1 = y_pred
            else:
                y_pred1 = np.concatenate((y_pred1,y_pred),axis=0)

        print(np.shape(y_pred1), ",",datanum)
        print(y_pred1)
        trade_data.out_indi_data(cfg.test_dataset,y_pred1,datalen=cfg.image_size)

if __name__ == "__main__":
    tf.app.run()