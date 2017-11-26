"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
from config import cfg
from utils import create_inputs

import capsnet_em as net

def main(_):
    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    with tf.Graph().as_default():
        num_batches_per_epoch_train = int(60000/cfg.batch_size)
        num_batches_test = int(10000/cfg.batch_size)

        batch_x, batch_labels = create_inputs(is_train=False)
        output = net.build_arch(batch_x, coord_add, is_train=False)
        batch_acc = net.test_accuracy(output, batch_labels)
        saver = tf.train.Saver()

        step = 0

        summaries = []
        summaries.append(tf.summary.scalar('accuracy', batch_acc))
        summary_op = tf.summary.merge(summaries)

        with tf.Session() as sess:
            tf.train.start_queue_runners(sess=sess)
            ckpt = tf.train.get_checkpoint_state(cfg.logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            summary_writer = tf.summary.FileWriter(cfg.test_logdir, graph=sess.graph)

            for epoch in range(cfg.test_epoch):

                accuracy_sum = 0
                for i in range(num_batches_test):
                    batch_acc_v, summary_str = sess.run([batch_acc, summary_op])
                    print('%d batches are tested.' % step)
                    summary_writer.add_summary(summary_str, step)
                    accuracy_sum += batch_acc_v
                    step += 1

                ave_acc = accuracy_sum/num_batches_test
                print(epoch,'epoch: average accuracy is %f' % ave_acc)

if __name__ == "__main__":
    tf.app.run()