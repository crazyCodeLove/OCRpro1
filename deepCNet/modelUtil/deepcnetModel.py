import tensorflow as tf

def conv2fc(inputs):
    conv_out_shape = inputs.get_shape().as_list()
    fcl_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    fcl_inputs = tf.reshape(inputs, [-1, fcl_in_features])
    return fcl_inputs,fcl_in_features

class DeepCModel(object):

    def __init__(self,hps,images,labels, mode, init_step):
        self.hps = hps
        self._images = images
        self.labes = labels
        self.mode = mode
        self.activateFunction=tf.nn.relu
        self.init_step = init_step

    def create_resnet(self, inputx, in_training, activateFunc=tf.nn.relu):
        # inputs:96

        cl1_kernal = 3
        cl1_in_depth = 9
        cl1_out_depth = self.hps.deep_net_fkn*1
        cl1_layer_name = "layer1"
        with tf.variable_scope(cl1_layer_name):
            # conv1:94
            conv1_weight = tf.get_variable('weight',
                                           [cl1_kernal, cl1_kernal, cl1_in_depth, cl1_out_depth],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # conv1_biases = tf.get_variable('bias', [cl1_out_depth])
            conv1 = tf.nn.conv2d(inputx, conv1_weight, strides=[1, 1, 1, 1], padding='VALID')
            # layer1_conv = tf.nn.bias_add(conv1, conv1_biases)
            layer1_conv = conv1
            layer1_conv = tf.layers.batch_normalization(layer1_conv, axis=0, training=in_training)
            layer1_conv = activateFunc(layer1_conv)

        cl2_layer_name = 'layer2'
        # layer2_pool:47
        with tf.variable_scope(cl2_layer_name):
            layer2_pool = tf.nn.max_pool(layer1_conv, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        cl3_kernal = 2
        cl3_in_depth = cl1_out_depth
        cl3_out_depth = self.hps.deep_net_fkn * 2
        cl3_layer_name = "layer3"
        # layer3_conv:46
        with tf.variable_scope(cl3_layer_name):
            conv3_weight = tf.get_variable('weight',
                                           [cl3_kernal, cl3_kernal, cl3_in_depth, cl3_out_depth],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # conv3_biases = tf.get_variable('bias', [cl3_out_depth])
            conv3 = tf.nn.conv2d(layer2_pool, conv3_weight, strides=[1, 1, 1, 1], padding='VALID')
            # layer3_conv = tf.nn.bias_add(conv3, conv3_biases)
            layer3_conv = conv3
            layer3_conv = tf.layers.batch_normalization(layer3_conv, axis=0, training=in_training)
            layer3_conv = activateFunc(layer3_conv)

        cl4_layer_name = 'layer4'
        # layer4_pool:23
        with tf.variable_scope(cl4_layer_name):
            layer4_pool = tf.nn.max_pool(layer3_conv, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        cl5_kernal = 2
        cl5_in_depth = cl3_out_depth
        cl5_out_depth = self.hps.deep_net_fkn * 3
        cl5_layer_name = "layer5"
        # layer5_conv:22
        with tf.variable_scope(cl5_layer_name):
            conv5_weight = tf.get_variable('weight',
                                           [cl5_kernal, cl5_kernal, cl5_in_depth, cl5_out_depth],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # conv5_biases = tf.get_variable('bias', [cl5_out_depth])
            conv5 = tf.nn.conv2d(layer4_pool, conv5_weight, strides=[1, 1, 1, 1], padding='VALID')
            # layer5_conv = tf.nn.bias_add(conv5, conv5_biases)
            layer5_conv = conv5
            layer5_conv = tf.layers.batch_normalization(layer5_conv, axis=0, training=in_training)
            layer5_conv = activateFunc(layer5_conv)

        cl6_layer_name = 'layer6'
        # layer6_pool:11
        with tf.variable_scope(cl6_layer_name):
            layer6_pool = tf.nn.max_pool(layer5_conv, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        cl7_kernal = 2
        cl7_in_depth = cl5_out_depth
        cl7_out_depth = self.hps.deep_net_fkn * 4
        cl7_layer_name = "layer7"
        # layer7_conv:10
        with tf.variable_scope(cl7_layer_name):
            conv7_weight = tf.get_variable('weight',
                                           [cl7_kernal, cl7_kernal, cl7_in_depth, cl7_out_depth],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv7_biases = tf.get_variable('bias', [cl7_out_depth])
            conv7 = tf.nn.conv2d(layer6_pool, conv7_weight, strides=[1, 1, 1, 1], padding='VALID')
            # layer7_conv = tf.nn.bias_add(conv7, conv7_biases)
            layer7_conv = conv7
            layer7_conv = tf.layers.batch_normalization(layer7_conv, axis=0, training=in_training)
            layer7_conv = activateFunc(layer7_conv)

        cl8_layer_name = 'layer8'
        # layer8_pool:5
        with tf.variable_scope(cl8_layer_name):
            layer8_pool = tf.nn.max_pool(layer7_conv, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        cl9_kernal = 2
        cl9_in_depth = cl7_out_depth
        cl9_out_depth = self.hps.deep_net_fkn * 5
        cl9_layer_name = "layer9"
        # layer9_conv:4
        with tf.variable_scope(cl9_layer_name):
            conv9_weight = tf.get_variable('weight',
                                           [cl9_kernal, cl9_kernal, cl9_in_depth, cl9_out_depth],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv9_biases = tf.get_variable('bias', [cl9_out_depth])
            conv9 = tf.nn.conv2d(layer8_pool, conv9_weight, strides=[1, 1, 1, 1], padding='VALID')
            layer9_conv = tf.nn.bias_add(conv9, conv9_biases)
            # layer9_conv = tf.layers.batch_normalization(layer9_conv, axis=0, training=in_training)
            layer9_conv = activateFunc(layer9_conv)

        cl10_layer_name = 'layer10'
        # layer10_pool:2
        with tf.variable_scope(cl10_layer_name):
            layer10_pool = tf.nn.max_pool(layer9_conv, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        cl11_kernal = 2
        cl11_in_depth = cl9_out_depth
        cl11_out_depth = self.hps.deep_net_fkn * 6
        cl11_layer_name = "layer11"
        # layer11_conv:1
        with tf.variable_scope(cl11_layer_name):
            conv11_weight = tf.get_variable('weight',
                                           [cl11_kernal, cl11_kernal, cl11_in_depth, cl11_out_depth],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv11_biases = tf.get_variable('bias', [cl11_out_depth])
            conv11 = tf.nn.conv2d(layer10_pool, conv11_weight, strides=[1, 1, 1, 1], padding='VALID')
            layer11_conv = tf.nn.bias_add(conv11, conv11_biases)
            # layer11_conv = tf.layers.batch_normalization(layer11_conv, axis=0, training=in_training)
            layer11_conv = activateFunc(layer11_conv)

        print("final outputs shape:", layer11_conv.get_shape().as_list())
        return layer11_conv




    def create_graph(self):
        # create graph start
        self.is_training_ph = tf.placeholder(tf.bool)

        self.step = tf.Variable(self.init_step, trainable=False)


        resnet = self.create_resnet(self._images, self.is_training_ph, activateFunc=self.activateFunction)
        fcl1_inputs, fcl1_in_features = conv2fc(resnet)
        # set outputs features
        outputs_features = self.hps.num_classes
        outputs = tf.layers.dense(fcl1_inputs, outputs_features)

        correct_prediction = tf.equal(tf.argmax(outputs, axis=1), tf.argmax(self.labes, axis=1))
        train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('train acc', train_accuracy)

        self.predictions = tf.nn.softmax(logits=outputs)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=outputs, labels=self.labes))

        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

        if self.mode == "train":
            self.learning_rate = tf.placeholder(tf.float32)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss=self.loss,global_step=self.step)

        self.init = tf.global_variables_initializer()
