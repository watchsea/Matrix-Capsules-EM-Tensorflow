"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
from utils import create_inputs
import time
import numpy as np
import os
import capsnet_em as net

def main(_):
    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    coord_add = np.array(coord_add, dtype=np.float32)/28.

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = int(60000/cfg.batch_size)
        opt = tf.train.AdamOptimizer()

        batch_x, batch_labels = create_inputs(is_train=True)
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)

        m_op = tf.placeholder(dtype=tf.float32, shape=())
        with tf.device('/gpu:0'):
            with slim.arg_scope([slim.variable], device='/cpu:0'):
                output = net.build_arch(batch_x, coord_add, is_train=True)
                # loss = net.cross_ent_loss(output, batch_labels)
                loss = net.spread_loss(output, batch_labels, m_op)

            grad = opt.compute_gradients(loss)

        loss_name = 'spread_loss'

        summaries = []
        summaries.append(tf.summary.scalar(loss_name, loss))

        train_op = opt.apply_gradients(grad, global_step=global_step)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.epoch)

        #read snapshot
        # latest = os.path.join(cfg.logdir, 'model.ckpt-4680')
        # saver.restore(sess, latest)

        # restore from the check point
        ckpt = tf.train.get_checkpoint_state(cfg.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.split('-')[1])
            print(ckpt, ckpt.model_checkpoint_path, initial_step)
        else:
            initial_step =0

        summary_op = tf.summary.merge(summaries)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(cfg.logdir, graph=sess.graph)

        m = 0.2
        for step in range(cfg.epoch):
            for i in range(num_batches_per_epoch):
                tic = time.time()
                _, loss_value = sess.run([train_op, loss], feed_dict={m_op: m})
                print('%d/%d, %d/%d iteration is finished in ' % (step,cfg.epoch,i,num_batches_per_epoch) + '%f second' % (time.time()-tic) + ',loss: %f'% loss_value)
                # test1_v = sess.run(test2)

                # if np.isnan(loss_value):
                #     print('bbb')
                #  assert not np.isnan(np.any(test2_v[0])), 'a is nan'
                assert not np.isnan(loss_value), 'loss is nan'

                if step % 10 == 0:
                    summary_str = sess.run(summary_op, feed_dict={m_op: m})
                    summary_writer.add_summary(summary_str, initial_step+step)

            if step > 0:
                m += (0.9-0.2)/(cfg.epoch*0.6)
                if m > 0.9:
                    m = 0.9

            ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=initial_step+step)

if __name__ == "__main__":
    tf.app.run()
