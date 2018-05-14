from collections import namedtuple
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle

from mnistCompRes.mnisttool import imgUtil
from mnistCompRes.mnisttool import ModelUtilv3s1
from deepCNet.modelUtil import deepcnetModel

HParams = namedtuple('HParams',
                     'batch_nums, num_classes, deep_net_fkn,'
                     'img_depth, img_width, des_img_size')


logger = ModelUtilv3s1.MyLog('/home/allen/work/data/resultlog/deepcNet/deepcnet9.txt')
logDir = '/home/allen/work/data/resultlog/deepcNet/summary/deepcs96f50BN3'

mnist = input_data.read_data_sets("../MNIST_DATA/", one_hot=True)
save_file = "/home/allen/work/variableSave/deepcnet/deepcnet.ckpt"
peizhi_filename = "/home/allen/work/chengxu/OCR/OCRpro1/deepCNet/peizhi.xml"

peizhi_dict = {'lrn_rate':1e-3,
               'is_restore':False,
               'train_step':0,
               'max_test_acc':0}
train_writer = tf.summary.FileWriter(logDir)

def startTrain(hps, mode):
    tf.reset_default_graph()
    with open(peizhi_filename, mode='rb') as rfobj:
        peizhi = pickle.load(rfobj)

    xp = tf.placeholder(tf.float32, [hps.batch_nums, hps.des_img_size, hps.des_img_size, hps.img_depth])
    yp = tf.placeholder(tf.float32, [None, 10])
    model = deepcnetModel.DeepCModel(hps, xp, yp, mode, peizhi['train_step'])
    model.create_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()

        if not peizhi['is_restore']:
            sess.run(model.init)
        else:
            saver.restore(sess, save_file)

        base_step = sess.run(model.step)
        end_step = int(base_step + 10000 + 1)
        for itstep in range(base_step,end_step):
            if (itstep%10) <= 8:
                images,labels = mnist.train.next_batch(hps.batch_nums)
            else:
                images,labels = mnist.validation.next_batch(hps.batch_nums)

            images = batch_imgs_process(images,hps)

            feed_dict = {
                xp:images,
                yp:labels,
                model.learning_rate: peizhi['lrn_rate'],
                model.is_training_ph: True}
            (inlabels,outprediction,cost,_ ) = sess.run(
                [model.labes, model.predictions, model.loss, model.train_op],
                feed_dict=feed_dict)

            if itstep % 100 == 0:
                (inlabels, outprediction, cost, _, summary) = sess.run(
                    [model.labes, model.predictions, model.loss, model.train_op, model.merged],
                    feed_dict=feed_dict)
                trainacc = ModelUtilv3s1.get_accurate(outprediction, inlabels)
                msg = "trainstep:%5d  loss:%e  train acc:%.5f"%(itstep, cost, trainacc)
                logger.log_message(msg)
                train_writer.add_summary(summary, itstep)

                if itstep % 500 == 0:
                    logger.showAndLogMsg(msg)


        print("before save")

        saver.save(sess=sess,save_path=save_file)
        print("after save")
        ModelUtilv3s1.update_peizhi(peizhi_filename,'is_restore',True)
        end_step = sess.run(model.step)
        ModelUtilv3s1.update_peizhi(peizhi_filename,'train_step',end_step)

def startTest(hps, mode):
    tf.reset_default_graph()
    with open(peizhi_filename, mode='rb') as rfobj:
        peizhi = pickle.load(rfobj)
    lrn_rate = peizhi['lrn_rate']

    xp = tf.placeholder(tf.float32, [hps.batch_nums, hps.des_img_size, hps.des_img_size, hps.img_depth])
    yp = tf.placeholder(tf.float32, [None, 10])

    model = deepcnetModel.DeepCModel(hps, xp, yp, mode, peizhi['train_step'])
    model.create_graph()

    # 训练一遍需要的步数
    epochTrainNums = int(10000 // hps.batch_nums)
    allrightnums = 0
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()

        saver.restore(sess=sess, save_path=save_file)

        for itstep in range(0, epochTrainNums):
            images, labels = mnist.test.next_batch(hps.batch_nums)
            images = batch_imgs_process(images,hps)

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
        ModelUtilv3s1.update_peizhi(peizhi_filename,'lrn_rate',lrn_rate)

        if test_acc > peizhi['max_test_acc']:
            ModelUtilv3s1.update_peizhi(peizhi_filename,'max_test_acc',test_acc)


        msg = "test acc:%.5f         now learning rate:%f"%(test_acc,lrn_rate)

        logger.showAndLogMsg(msg)

def batch_imgs_process(batch_imgs, hps):
    batch_imgs = imgUtil.batch_imgs_reshape(batch_imgs, hps)
    batch_imgs = imgUtil.batch_imgs_resize(batch_imgs, hps.des_img_size, hps)
    return batch_imgs


def main():
    hps = HParams(batch_nums=50,
                  num_classes=10,
                  deep_net_fkn=100,
                  img_depth=1,
                  img_width=28,
                  des_img_size=96
                  )
    ModelUtilv3s1.init_peizhi(
        peizhifilename=peizhi_filename, peizhidict=peizhi_dict)

    while True:
        print("start training")
        mode = 'train'
        startTrain(hps,mode=mode)
        print("training end")

        print("start test")
        mode = 'test'

        startTest(hps,mode=mode)

        print("test end")

        with open(peizhi_filename, mode='rb') as rfobj:
            peizhi = pickle.load(rfobj)
        if peizhi['max_test_acc'] >= 0.999:
            print("already over best test acc, now test acc is ", peizhi['max_test_acc'])
            break

if __name__ == "__main__":
    main()