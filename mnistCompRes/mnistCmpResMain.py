#!/usr/bin/env python
#coding=utf-8
import pickle
from collections import namedtuple

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnistCompRes.mnisttool import ModelUtilv3s1
from mnistCompRes.mnisttool import imgUtil
from mnistCompRes.mnisttool import mnistModel
from tools import coreMailUtil

"""
weight layer number is 2+2*carriage_block_num + len(carriage_block_num)=2*[2,2,2,2]+ 6= 22

"""

HParams = namedtuple('HParams',
                     'batch_nums, num_classes, deep_net_fkn,'
                     'img_depth, img_width, deepk, carriage_block_num,'
                     'des_img_size, descrate')

GParams = namedtuple('GParams',
                     'save_file_name, des_save_dirname, save_dirname, peizhi_filename')

peizhi_dict = {'lrn_rate':1e-2,
               'is_restore':False,
               'train_step':0,
               'test_step':0,
               'max_test_acc':0}


logger = ModelUtilv3s1.MyLog('/home/allen/work/data/resultlog/mnistCompRes/mnistCmpResn9.txt')
mnist = input_data.read_data_sets("../MNIST_DATA/", one_hot=True)


def startTrain(trainepochnums, hps, mode, gps):
    # images,labels = mnist.train.next_batch(hps.batch_nums)
    tf.reset_default_graph()

    xp = tf.placeholder(tf.float32, [None, hps.des_img_size, hps.des_img_size, hps.img_depth])
    yp = tf.placeholder(tf.float32, [None, 10])
    model = mnistModel.MnistModel(hps, xp, yp, mode)
    model.create_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        with open(gps.peizhi_filename, mode='rb') as rfobj:
            peizhi = pickle.load(rfobj)

        if not peizhi['is_restore']:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, gps.save_file_name)

        base_step = int(peizhi['train_step'])
        end_step = int(base_step + 50000*trainepochnums / hps.batch_nums +1)
        # end_step = int(base_step + 5000*trainepochnums / hps.batch_nums +1)
        for itstep in range(base_step,end_step):
            if (itstep%10) <= 8:
                images,labels = mnist.train.next_batch(hps.batch_nums)
            else:
                images,labels = mnist.validation.next_batch(hps.batch_nums)

            images = batch_imgs_process_train(images,hps)

            feed_dict = {
                xp:images,
                yp:labels,
                model.learning_rate: peizhi['lrn_rate'],
                model.is_training_ph: True}
            (inlabels,outprediction,cost,_ ) = sess.run(
                [model.labes, model.predictions, model.loss, model.train_op],
                feed_dict=feed_dict)

            if itstep % 100 == 0:
                trainacc = ModelUtilv3s1.get_accurate(outprediction, inlabels)
                msg = "trainstep:%5d  loss:%e  train acc:%.5f"%(itstep, cost, trainacc)

                if itstep % 200 ==0:
                    logger.showAndLogMsg(msg)
                else:
                    logger.log_message(msg)

            # if itstep % 7000 ==0 and itstep > 0:
            #     print "before save"
            #     saver.save(sess=sess, save_path=gps.save_file_name)
            #     print "after save"

        print("before save")
        saver.save(sess=sess,save_path=gps.save_file_name)
        print("after save")
        ModelUtilv3s1.update_peizhi(gps.peizhi_filename,'is_restore',True)
        ModelUtilv3s1.update_peizhi(gps.peizhi_filename,'train_step',end_step)





def startTest(hps, mode, gps, msg):
    # images, labels = mnist.test.next_batch(hps.batch_nums)
    tf.reset_default_graph()

    xp = tf.placeholder(tf.float32, [None, hps.des_img_size, hps.des_img_size, hps.img_depth])
    yp = tf.placeholder(tf.float32, [None, 10])

    model = mnistModel.MnistModel(hps, xp, yp, mode)
    model.create_graph()

    # 训练一遍需要的步数
    epochTrainNums = int(10000 // hps.batch_nums)
    allrightnums = 0
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()

        saver.restore(sess=sess, save_path=gps.save_file_name)

        with open(gps.peizhi_filename,mode='rb') as rfobj:
            peizhi = pickle.load(rfobj)
            lrn_rate = peizhi['lrn_rate']

        base_step = peizhi['test_step']
        end_step = int(base_step + epochTrainNums)
        for itstep in range(base_step, end_step):
            images, labels = mnist.test.next_batch(hps.batch_nums)
            images = batch_imgs_process_test(images,hps)

            feed_dict = {
                xp: images,
                yp: labels,
                model.is_training_ph: False}
            (inlabels, outprediction) = sess.run(
                [model.labes, model.predictions],
                feed_dict=feed_dict)
            itrightnums = ModelUtilv3s1.get_test_right_num(outprediction, inlabels)
            allrightnums += itrightnums


        test_acc = float(allrightnums)/10000
        lrn_rate = ModelUtilv3s1.down_learning_rate(test_acc,lrn_rate)
        ModelUtilv3s1.update_peizhi(gps.peizhi_filename,'lrn_rate',lrn_rate)
        ModelUtilv3s1.update_peizhi(gps.peizhi_filename,'test_step',end_step)

        if test_acc > peizhi['max_test_acc']:
            ModelUtilv3s1.update_peizhi(gps.peizhi_filename,'max_test_acc',test_acc)
            ModelUtilv3s1.move_variable_from_src2des(gps.save_dirname, gps.des_save_dirname)


            email = coreMailUtil.Email('xxx','xxx@')
            to_list = ['xxx@163.com']
            content = msg + "<p>max_acc:"+str(test_acc)+"</p>"
            email.send_mail_html(to_list, 'OCR latest acc',content)


        msg = "test acc:%.5f         now learning rate:%f"%(test_acc,lrn_rate)

        logger.showAndLogMsg(msg)


def batch_imgs_process_train(batch_imgs, hps):
    batch_imgs = imgUtil.batch_imgs_reshape(batch_imgs, hps)
    batch_imgs = imgUtil.batch_imgs_resize(batch_imgs, hps.des_img_size, hps)
    batch_imgs = imgUtil.batch_imgs_aug(batch_imgs)
    return batch_imgs


def batch_imgs_process_test(batch_imgs, hps):
    batch_imgs = imgUtil.batch_imgs_reshape(batch_imgs, hps)
    batch_imgs = imgUtil.batch_imgs_resize(batch_imgs, hps.des_img_size, hps)
    return batch_imgs




def batch_imgs_reshape_and_agu(batch_imgs,hps):
    """
    将images数据shape由 (batch,hps.img_width*hps.img_width*hps.img_depth) -> (batch, hps.img_width, hps.img_width, hps.img_depth)
    并且进行数据增强
    :param batch_imgs:
    :param hps:
    :return:
    """
    batch_imgs = imgUtil.batch_imgs_reshape(batch_imgs,hps)
    batch_imgs = imgUtil.batch_imgs_aug(batch_imgs)
    return batch_imgs


def main():
    hps = HParams(batch_nums=50,
                  num_classes=10,
                  deep_net_fkn=24,
                  img_depth=1,
                  img_width=28,
                  deepk=[2.2, 2.2, 2.2, 2.1],
                  carriage_block_num=[2,2,2,2],
                  des_img_size=96,
                  descrate=[0.95, 0.95, 0.95, 0.95])

    save_file_name = '/home/allen/work/variableSave/OCRpro1/temp/deepres.ckpy'
    des_save_dirname = '/home/allen/work/variableSave/OCRpro1/mnist/mnistCompRes'
    save_dirname = '/home/allen/work/variableSave/OCRpro1/temp/'

    gps = GParams(save_file_name=save_file_name,
                  des_save_dirname=des_save_dirname,
                  save_dirname=save_dirname,
                  peizhi_filename='/home/allen/work/chengxu/OCR/OCRpro1/mnistCompRes/peizhi.xml'
                  )

    ModelUtilv3s1.init_peizhi(
        peizhifilename=gps.peizhi_filename, peizhidict=peizhi_dict)

    msg = "peizhi\nhps:"+str(hps) +\
          "\ngps:"+str(gps)+ \
          "\n max pool"+\
          "deepk and descrate can be fractional, descrate=0.6  use Relu, "+ \
          "\nimg random crop scale rotate"
    logger.showAndLogMsg(msg)

    while True:
        print("start training")
        mode = 'train'
        trainNumsBeforeValid = 4
        startTrain(trainNumsBeforeValid,hps,mode=mode,gps=gps)
        print("training end")

        print("start test")
        mode = 'test'

        startTest(hps,mode=mode, gps=gps, msg=msg)

        print("test end")

        with open(gps.peizhi_filename, mode='rb') as rfobj:
            peizhi = pickle.load(rfobj)
        if peizhi['max_test_acc'] >= 0.999:
            print("already over best test acc, now test acc is ", peizhi['max_test_acc'])
            break


if __name__ == "__main__":
    main()