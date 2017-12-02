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
        batch_x, batch_labels, datanum = utils.get_batch_data(is_training=False)
        num_batches_test = math.ceil(datanum / cfg.batch_size)     #get the ceiling int

        output = net.build_arch(batch_x, coord_add, is_train=False)
        predict = tf.argmax(output, axis=1)
        batch_acc = net.test_accuracy(output, batch_labels)
        saver = tf.train.Saver()

        step = 0

        summaries = []
        summaries.append(tf.summary.scalar('accuracy', batch_acc))
        summary_op = tf.summary.merge(summaries)

        sess = tf.Session()
        tf.train.start_queue_runners(sess=sess)
        ckpt = tf.train.get_checkpoint_state(cfg.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(cfg.test_logdir, graph=sess.graph)

        for epoch in range(cfg.test_epoch):
            accuracy_sum = 0
            for i in range(num_batches_test):
                y_pred,y,batch_acc_v, summary_str = sess.run([predict,batch_labels,batch_acc, summary_op])

                if i% 10 ==0:
                    print('%d/%d batches are tested.' % (step,num_batches_test))
                    #print("labels:\n",batch_labels)
                    print("Y:\n",y)
                    print("Y_prediction:",batch_acc_v,"\n",y_pred)
                summary_writer.add_summary(summary_str, step)
                accuracy_sum += batch_acc_v
                step += 1
                if i==0:
                    y_pred1 = y_pred
                else:
                    y_pred1 = np.concatenate((y_pred1,y_pred),axis=0)



            ave_acc = accuracy_sum/num_batches_test
            # print("The last batch----Y:",np.shape(y),"\n", y)
            # print("Y_prediction:", batch_acc_v, "\n", y_pred)
            print(epoch,'epoch: average accuracy is %f' % ave_acc)
            print(np.shape(y_pred1), ",",datanum)
            trade_data.out_indi_data(cfg.test_dataset,y_pred1)

if __name__ == "__main__":
    tf.app.run()