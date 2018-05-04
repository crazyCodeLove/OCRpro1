#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnistCompRes.mnisttool import ModelUtilv3s1

#create data
mnist = input_data.read_data_sets("../MNIST_DATA/", one_hot=True)
mode = 'train'
mode = 'test'


logger = ModelUtilv3s1.MyLog('/home/allen/work/data/resultlog/lenet5/lenet2.txt')

#set global features
img_size = 28
img_depth = 1

keep_prob = tf.placeholder(tf.float32)
train_nums = 10000
batch_size = 50

save_file = "/home/allen/work/variableSave/lenet5/lenet1.ckpt"
learning_rate = 1e-4

activate_func = tf.nn.relu

#create graph
    #set conv layer 1 features
xp = tf.placeholder(tf.float32,[None,img_size*img_size*img_depth])
xp_reshape = tf.reshape(xp,[-1,img_size,img_size,img_depth])

yp = tf.placeholder(tf.float32,[None,10])

cl1_kernal = 5
cl1_in_depth = img_depth
cl1_out_depth = 6
cl1_layer_name = "layer1"
#conv_layer1 is 28*28*10
with tf.variable_scope(cl1_layer_name):
    conv1_weight = tf.get_variable('weight',
                                   [cl1_kernal,cl1_kernal,cl1_in_depth,cl1_out_depth],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv1_biases = tf.get_variable('bias',[cl1_out_depth])
    conv1 = tf.nn.conv2d(xp_reshape,conv1_weight,strides=[1,1,1,1],padding='SAME')
    layer1_conv = tf.nn.bias_add(conv1,conv1_biases)

cl2_layer_name='layer2'
cl2_kernal = 2
# 14*14*6
with tf.variable_scope(cl2_layer_name):
    layer2_pool = tf.nn.avg_pool(layer1_conv,[1,cl2_kernal,cl2_kernal,1],strides=[1,2,2,1],padding='SAME')
    layer2_pool = activate_func(layer2_pool)


cl3_kernal = 5
cl3_in_depth = cl1_out_depth
cl3_out_depth = 16
cl3_layer_name = "layer3"
# 10*10*16
with tf.variable_scope(cl3_layer_name):
    conv3_weight = tf.get_variable('weight',
                                   [cl3_kernal,cl3_kernal,cl3_in_depth,cl3_out_depth],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv3_biases = tf.get_variable('bias',[cl3_out_depth])
    conv3 = tf.nn.conv2d(layer2_pool,conv3_weight,strides=[1,1,1,1],padding='VALID')
    layer3_conv = tf.nn.bias_add(conv3,conv3_biases)


cl4_layer_name='layer4'
cl4_kernal = 2
# 5*5*16
with tf.variable_scope(cl4_layer_name):
    layer4_pool = tf.nn.avg_pool(layer3_conv,[1,cl4_kernal,cl4_kernal,1],strides=[1,2,2,1],padding='SAME')
    layer4_pool = activate_func(layer4_pool)

cl5_kernal = 5
cl5_in_depth = cl3_out_depth
cl5_out_depth = 120
cl5_layer_name = "layer5"
# 1*1*120
with tf.variable_scope(cl5_layer_name):
    conv5_weight = tf.get_variable('weight',
                                   [cl5_kernal,cl5_kernal,cl5_in_depth,cl5_out_depth],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv5_biases = tf.get_variable('bias',[cl5_out_depth])
    conv5 = tf.nn.conv2d(layer4_pool,conv5_weight,strides=[1,1,1,1],padding='VALID')
    layer5_conv = tf.nn.bias_add(conv5,conv5_biases)




#reshape convolution layer to full connected layer feature
conv_out_shape = layer5_conv.get_shape().as_list()
fcl1_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]

fcl1_inputs = tf.reshape(layer5_conv,[-1,fcl1_in_features])
fcl1_out_features = 84

fc_layer1 = tf.layers.dense(inputs=fcl1_inputs,units=fcl1_out_features,activation=tf.nn.relu)
#set output layer features
outputs_features = 10
outputs = tf.layers.dense(inputs=fc_layer1,units=outputs_features)

loss = tf.losses.softmax_cross_entropy(onehot_labels=yp,logits=outputs)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()


if mode == 'train':
    with tf.Session() as sess:
        # sess.run(init)
        saver.restore(sess, save_file)
        for i in range(train_nums):
            train_x, train_y = mnist.train.next_batch(batch_size)
            lo, ou, _ = sess.run([loss, outputs, train_step], feed_dict={xp: train_x, yp: train_y, keep_prob: 1})

            if i % 100 == 0:
                trainacc = ModelUtilv3s1.get_accurate(ou, train_y)
                msg = "trainstep:%5d  loss:%e  train acc:%.5f" % (i, lo, trainacc)
                logger.showAndLogMsg(msg)

        saver.save(sess, save_file)

else:
    allrightnums = 0
    with tf.Session() as sess:
        saver.restore(sess, save_file)

        for i in range(200):
            test_x, test_y = mnist.test.next_batch(50)
            outs = sess.run(outputs, feed_dict={xp: test_x, yp: test_y, keep_prob: -1})

            itrightnums = ModelUtilv3s1.get_test_right_num(outs, test_y)
            allrightnums += itrightnums

        test_acc = float(allrightnums) / 10000
    msg = "test acc:%.5f" % (test_acc)

    logger.showAndLogMsg(msg)

