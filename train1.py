"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from config import cfg
import utils
import time
import numpy as np
import os
import sys
import capsnet_em as net

def main(_):
    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    coord_add = np.array(coord_add, dtype=np.float32)/28.
    data = utils.load_trade(is_training=True)
    datanum = data.num_examples
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        batch_x =tf.placeholder(tf.float32,[cfg.batch_size,cfg.image_size,cfg.image_size,3])
        batch_labels = tf.placeholder(tf.int32,[cfg.batch_size])
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        opt = tf.train.AdamOptimizer()

        #batch_x, batch_labels,datanum = utils.get_shuffle_batch_data(is_training=True)
        num_batches_per_epoch = int(datanum / cfg.batch_size)
        print(datanum,num_batches_per_epoch)
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)

        m_op = tf.placeholder(dtype=tf.float32, shape=())
        with tf.device('/gpu:0'):
            with slim.arg_scope([slim.variable], device='/cpu:0'):
                output = net.build_arch(batch_x, coord_add, is_train=True)
                # loss = net.cross_ent_loss(output, batch_labels)
                loss = net.spread_loss(output, batch_labels, m_op)
                accuracy = net.test_accuracy(output,batch_labels)
            grad = opt.compute_gradients(loss)

        loss_name = 'spread_loss'

        # Print trainable variable parameter statistics to stdout.
        # By default, statistics are associated with each graph node.
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        # param_stats is tensorflow.tfprof.TFGraphNodeProto proto.
        # Let's print the root below.
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        summaries = []
        summaries.append(tf.summary.scalar(loss_name, loss))
        summaries.append(tf.summary.scalar("accuracy",accuracy))

        train_op = opt.apply_gradients(grad, global_step=global_step)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        # add addition options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)  #cfg.epoch)

        # restore from the check point
        ckpt = tf.train.get_checkpoint_state(cfg.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.split('-')[1])
            print(ckpt, ckpt.model_checkpoint_path, initial_step)
        else:
            initial_step =0
        m = 0.2

        summary_op = tf.summary.merge(summaries)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(cfg.logdir, graph=sess.graph)


        cal_num=0
        for step in range(cfg.epoch):
            for i in range(num_batches_per_epoch):
                tic = time.time()
                x,y = data.next_batch(cfg.batch_size)
                _, loss_value,accuracy_val = sess.run([train_op, loss,accuracy], feed_dict={batch_x:x,batch_labels:y,m_op: m})
                print('%d/%d, %d/%d iteration is finished in ' % (step,cfg.epoch,i,num_batches_per_epoch) + '%f second' % (time.time()-tic) + ',m:',m,',loss: %f'% loss_value,",accuracy:",accuracy_val)

                assert not np.isnan(loss_value), 'loss is nan'
                cal_num+=1
                if i % 30 == 0:

                    summary_str = sess.run(summary_op, feed_dict={batch_x:x,batch_labels:y,m_op: m},
                                           options=options,
                                           run_metadata=run_metadata
                                           )
                    summary_writer.add_run_metadata(run_metadata,'step%d'% cal_num)
                    summary_writer.add_summary(summary_str, initial_step+cal_num)

                    # Print to stdout an analysis of the memory usage and the timing information
                    # broken down by operations.
                    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
                    #     tf.get_default_graph(),
                    #     run_meta=run_metadata,
                    #     tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

                    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    # with open('./time_line/timeline_02_step_%d.json' % i, 'w') as f:
                    #     f.write(chrome_trace)

                if cal_num % cfg.saveperiod == 0:
                    ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
                    saver.save(sess, ckpt_path, global_step=initial_step + cal_num)

                if m<0.9:
                    m += round((0.9-0.2) / num_batches_per_epoch,5)
                else:
                    m = 0.9

            ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=initial_step+cal_num)

if __name__ == "__main__":
    tf.app.run()
