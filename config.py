import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################
flags.DEFINE_float('ac_lambda0', 1.0, '\lambda in the activation function a_c, iteration 0')
flags.DEFINE_float('ac_lambda_step', 1.0, 'It is described that \lambda increases at each iteration with a fixed schedule, however specific super parameters is absent.')

flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('epoch', 200, 'epoch')
flags.DEFINE_integer('test_epoch',1,'test epoch')
flags.DEFINE_integer('iter_routing', 1, 'number of iterations')
flags.DEFINE_integer('saveperiod',500,'period of save model')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')

################################
#    structure parameters      #
################################
flags.DEFINE_integer('A', 32, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'E:/study/dl/train_data/*.0000.*.csv', 'the path for train dataset')   #data/mnist
flags.DEFINE_string('test_dataset', 'E:/study/dl/test_data/rb.HOT.60m(20171202).csv', 'the path for test dataset,must be only one')   #data/mnist
flags.DEFINE_string('indi_path',"indi_out/", 'the indicators store path')
flags.DEFINE_string('label_path',"label_out/", 'the true label data store path')
flags.DEFINE_boolean('is_train', True, 'train or predict phase')
flags.DEFINE_boolean('is_true_out', False, 'ouput the true label data')
flags.DEFINE_boolean('is_feature_store', False, 'ouput the true label data')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('test_logdir', 'test_logdir', 'test logs directory')

############################
#   application setting    #
############################
flags.DEFINE_integer('image_size',28,'image size')
flags.DEFINE_integer('label_num',19,'laber number')
flags.DEFINE_integer('data_period',125,'sample data period')
flags.DEFINE_integer('label_post_num',20,'the post period of data to calculate the profit')
flags.DEFINE_float('loss_ratio',0.05,'the maximum loss ratio')
flags.DEFINE_float('profit_ratio', 0.20, 'the minium profit ratio')

cfg = tf.app.flags.FLAGS