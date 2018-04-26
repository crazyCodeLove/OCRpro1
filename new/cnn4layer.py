#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#create data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

#define add conv layer function
def add_conv_layer(inputs, patch_size, in_depth, out_depth, layer_name="layer", activate_function=None, keep_prob=-1, strides = [1,1,1,1]):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,in_depth, out_depth],stddev=0.1))
        biases = tf.Variable(tf.constant(0.1,tf.float32,[out_depth]))

        y = tf.nn.conv2d(inputs, Weights,strides,padding="SAME") + biases

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        if keep_prob!=-1:
            outputs = tf.nn.dropout(outputs,keep_prob=keep_prob)

        return outputs

def add_max_pool(inputs, step):
    kernal = [1,step,step,1]
    return tf.nn.max_pool(inputs,ksize=kernal,strides=kernal,padding="SAME")

def add_fc_layer(inputs, in_features, out_features, layer_name="layer", activate_function=None, keep_prob=-1):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.truncated_normal([in_features,out_features],stddev=0.1))
        biases = tf.Variable(tf.constant(0.1,tf.float32,[out_features]))
        y = tf.matmul(inputs,Weights) + biases

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        if keep_prob != -1:
            outputs = tf.nn.dropout(outputs,keep_prob)

        return outputs,Weights

def get_accuracy(prediction,labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.arg_max(labels,1)),dtype=tf.float32))



#set global features
img_size = 28
img_depth = 1

keep_prob = tf.placeholder(tf.float32)
train_nums = 80000000001
batch_size = 100

save_file = "/home/allen/work/variableSave/cnn/cnn4layer.ckpt"
learning_rate = 1e-4

alpha = 1e-3

#create graph
    #set conv layer 1 features
xp = tf.placeholder(tf.float32,[None,img_size*img_size*img_depth])
xp_reshape = tf.reshape(xp,[-1,img_size,img_size,img_depth])

yp = tf.placeholder(tf.float32,[None,10])

cl1_patch_size = 3
cl1_in_depth = img_depth
cl1_out_depth = 30
cl1_layer_name = "conv1"

#conv_layer1 is 28*28*10
conv_layer1 = add_conv_layer(xp_reshape,cl1_patch_size,cl1_in_depth,cl1_out_depth,activate_function=tf.nn.relu)

    #set conv layer 2 features
cl2_patch_size = 2
cl2_out_depth = 60
#conv layer2 is 28*28*32
conv_layer2 = add_conv_layer(conv_layer1,cl2_patch_size,cl1_out_depth,cl2_out_depth,activate_function=tf.nn.relu)

#pooling layer 1 is 14*14*32
pool_layer1 = add_max_pool(conv_layer2,step=2)

    #set convolution layer 3 features
cl3_patch_size = 2
cl3_out_depth = 90
#conv_layer3 is 14*14*64
conv_layer3 = add_conv_layer(pool_layer1,cl3_patch_size,cl2_out_depth,cl3_out_depth,keep_prob=keep_prob,activate_function=tf.nn.relu)

    #set convolution 4 features
cl4_patch_size = 2
cl4_out_depth = 120
#conv_layer4 is 14*14*128
conv_layer4 = add_conv_layer(conv_layer3,cl4_patch_size,cl3_out_depth,cl4_out_depth,keep_prob=keep_prob,activate_function=tf.nn.relu)

#pool_layer2 is 7*7*128
pool_layer2 = add_max_pool(conv_layer4,step=2)





#reshape convolution layer to full connected layer feature
conv_out_shape = pool_layer2.get_shape().as_list()
fcl1_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]

fcl1_inputs = tf.reshape(pool_layer2,[-1,fcl1_in_features])
fcl1_out_features = 1024

fc_layer1,Weights1 = add_fc_layer(fcl1_inputs,fcl1_in_features,fcl1_out_features,activate_function=tf.nn.relu,keep_prob=keep_prob)

#set fully connected layer 2 features
fcl2_out_features = 512
fc_layer2,Weights2 = add_fc_layer(fc_layer1,fcl1_out_features,fcl2_out_features,activate_function=tf.nn.relu,keep_prob=keep_prob)

#set output layer features
outputs_features = 10
outputs,Weights3 = add_fc_layer(fc_layer2,fcl2_out_features,outputs_features,keep_prob=keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=yp)) + \
       alpha*(tf.nn.l2_loss(Weights1) + tf.nn.l2_loss(Weights2) + tf.nn.l2_loss(Weights3))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()


with tf.Session() as sess:
    # sess.run(init)
    saver.restore(sess,save_file)

    for i in range(train_nums):
        train_x,train_y = mnist.train.next_batch(batch_size)
        lo,ou,_ = sess.run([loss,outputs,train_step],feed_dict={xp:train_x, yp:train_y, keep_prob:0.8})



        if i % 200 == 0 :
            # test_x = mnist.test.images
            # test_y = mnist.test.labels

            test_x,test_y = mnist.test.next_batch(2000)
            outs = sess.run(outputs,feed_dict={xp:test_x, yp:test_y, keep_prob:-1})
            acc = sess.run(get_accuracy(outs, test_y))
            print("step %10d loss %2.5f acc %.5f" % (i, lo, acc))

        if i%1000 == 0:
            saver.save(sess, save_file)




