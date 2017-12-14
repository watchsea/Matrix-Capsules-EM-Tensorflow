"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import utils
import time
import numpy as np
import os
import capsnet_em as net

def main(_):
    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    coord_add = np.array(coord_add, dtype=np.float32)/28.

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        batch_x, batch_labels = utils.get_shuffle_tfrecord(is_training=True)
        datanum = 272965
        num_batches_per_epoch = int(datanum / cfg.batch_size)
        print(datanum,num_batches_per_epoch)
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)
        """Use exponential decay leanring rate?"""
        lrn_rate = tf.maximum(tf.train.exponential_decay(
            1e-3, global_step, num_batches_per_epoch, 0.8), 1e-5)
        tf.summary.scalar('learning_rate', lrn_rate)
        opt = tf.train.AdamOptimizer(learning_rate=lrn_rate)

        m_op = tf.placeholder(dtype=tf.float32, shape=())
        with tf.device('/gpu:0'):
            with slim.arg_scope([slim.variable], device='/cpu:0'):
                output = net.build_arch(batch_x, coord_add, is_train=True)
                # loss = net.cross_ent_loss(output, batch_labels)
                loss = net.spread_loss(output, batch_labels, m_op)
                accuracy = net.test_accuracy(output,batch_labels)
                tf.summary.scalar("spread_loss", loss)
                tf.summary.scalar("accuracy", accuracy)

            """Compute gradient."""
            grad = opt.compute_gradients(loss)
            # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
            grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                          for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

        """Apply graident."""
        with tf.control_dependencies(grad_check):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(grad, global_step=global_step)

            # Print trainable variable parameter statistics to stdout.
            # By default, statistics are associated with each graph node.
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        # param_stats is tensorflow.tfprof.TFGraphNodeProto proto.
        # Let's print the root below.
        print('total_params: %d\n' % param_stats.total_parameters)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=30)  #cfg.epoch)

        # restore from the check point
        ckpt = tf.train.get_checkpoint_state(cfg.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.split('-')[1])
            print(ckpt, ckpt.model_checkpoint_path, initial_step)
            m =0.9
        else:
            initial_step =0
            m = 0.2

        # read snapshot
        # latest = os.path.join(cfg.logdir, 'model.ckpt-4680')
        # saver.restore(sess, latest)
        """Set summary op."""
        summary_op = tf.summary.merge_all()

        """Start coord & queue."""
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        """Set summary writer"""
        # if not os.path.exists(cfg.logdir):
        #     os.makedirs(cfg.logdir)
        summary_writer = tf.summary.FileWriter(
            cfg.logdir , graph=sess.graph)  # graph = sess.graph, huge!

        cal_num=0

        for step in range(cfg.epoch):
            for i in range(num_batches_per_epoch):
                tic = time.time()
                """"TF queue would pop batch until no file"""
                try:
                    _, loss_value,accuracy_val = sess.run([train_op, loss,accuracy], feed_dict={m_op: m})
                    print('%d/%d, %d/%d iteration is finished in ' % (step,cfg.epoch,i,num_batches_per_epoch) + '%f second' % (time.time()-tic) + ',m:',m,',loss: %f'% loss_value,",accuracy:",accuracy_val)

                    cal_num += 1
                except tf.errors.InvalidArgumentError:
                    print('%d iteration contains NaN gradients. Discard.' % cal_num)
                    continue
                else:
                    """Write to summary."""
                    if i % 30 == 0:
                        summary_str = sess.run(summary_op, feed_dict={m_op: m})
                        summary_writer.add_summary(summary_str, initial_step+cal_num)

                    if cal_num % cfg.saveperiod == 0:
                        ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
                        saver.save(sess, ckpt_path, global_step=initial_step + cal_num)

                    if m<0.9:
                        m += round((0.9-0.2) / num_batches_per_epoch,5)
                    else:
                        m = 0.9

            ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=initial_step+cal_num)

        """Join threads"""
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
