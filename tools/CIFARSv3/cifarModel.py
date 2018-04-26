#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from tools.CIFARSv3 import ModelUtilv3s1


class CifarModel(object):


    def __init__(self,hps,images,labels, mode, logger):
        """
        进行初始化
        :param hps:
        :param images:
        :param labels:
        """
        self.hps = hps
        self._images = images
        self.labes = labels
        self.mode = mode
        self.logger = logger


    def create_deep_res_head(self, inputx, is_training_ph, activateFunc=tf.nn.relu):
        """
        会进行一次max pooling,
        一层卷积层,一层max pooling,一个building block serial组成
        """
        kernal_width = 3
        depth = self.hps.deep_net_fkn

        # conv_layer1 is 64*64*deep_net_fkn
        outputs = ModelUtilv3s1.add_BN_conv_layer(
            inputx, kernal_width, self.hps.img_depth,
            depth, is_training_ph, scope="reshead",
            activateFunc=activateFunc)

        return outputs

    def create_deep_res_body(
            self,batch_num, carriage_block_num, inputs, is_training_ph,
            layername="layer", activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
        """

        最后的输出是4*4*deep_net_fkn*(2**4)
        """
        kernalWidth = 3

        outputs = inputs
        for it in range(len(carriage_block_num)):
            if 0==it:
                outputs = ModelUtilv3s1.add_overlap_maxpool(outputs)
            else:
                outputs = ModelUtilv3s1.add_averagepool_layer(outputs)


            tscope = "carriage_" + str(it)
            outputs = ModelUtilv3s1.add_building_block_carriage(
                batch_num, self.hps.deepk[it], carriage_block_num[it], outputs,
                kernalWidth, is_training_ph,
                scope=tscope, layername=layername, activateFunc=activateFunc,
                stride=stride)
            tscope = "carriage" + str(it) + "des"
            outputs = ModelUtilv3s1.building_block_desc(
                outputs, is_training_ph, scope=tscope, layername=layername,
                activateFunc=None, stride=stride)
        return outputs

    def create_resnet(self, inputx, is_training_ph, activateFunc=tf.nn.relu):
        reshead = self.create_deep_res_head(
            inputx, is_training_ph, activateFunc=activateFunc)

        stride = [1,1,1,1]
        outputs = self.create_deep_res_body(
            self.hps.batch_nums, self.hps.carriage_block_num, reshead,
            is_training_ph, activateFunc=activateFunc,stride=stride)

        ##########################
        msg = "before fractional outputs shape:" + str(outputs.get_shape().as_list())
        self.logger.showAndLogMsg(msg)
        outputs = ModelUtilv3s1.add_fractional_maxpool(outputs, ratio=[1, 1.4, 1.4, 1])

        msg = "after fractional outputs shape:"+ str(outputs.get_shape().as_list())
        self.logger.showAndLogMsg(msg)
        ###########################

        return outputs

    def create_graph(self):
        # create graph start

        self.is_training_ph = tf.placeholder(tf.bool)

        resnet = self.create_resnet(self._images, self.is_training_ph, activateFunc=tf.nn.relu)
        fcl1_inputs, fcl1_in_features = ModelUtilv3s1.conv2fc(resnet)
        # set outputs features
        outputs_features = self.hps.num_classes
        outputs = ModelUtilv3s1.add_fc_layer(fcl1_inputs, fcl1_in_features,
                                                         outputs_features)

        self.predictions = tf.nn.softmax(logits=outputs)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=outputs, labels=self.labes))

        if self.mode == "train":
            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss=self.loss)

        
        
