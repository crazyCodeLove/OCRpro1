import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage import util
from skimage.morphology import disk
from skimage.filters import rank
from matplotlib import pyplot as plt
from skimage import filters

from PIL import Image

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

def get_single_img_label():
    for epoch in range(10000):
        images, labels = mnist.train.next_batch(10,shuffle=True)
        for it in range(10):
            img = np.reshape(images[it], newshape=(28, 28))
            yield (img,labels[it])

def resize_pic(img,des_size):
    img = Image.fromarray(img)
    return np.asanyarray(img.resize(des_size))

def gen_salt_filter():
    '''
    添加椒盐噪声然后使用中值滤波和均值滤波
    '''
    gen = get_single_img_label()
    for img,label in gen:
        img = util.random_noise(img, mode='salt')

        fil_camer = rank.median(img, selem=disk(3))
        media_camer = rank.mean(img, selem=disk(3))

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title('add salt noise')

        plt.subplot(2, 2, 2)
        plt.imshow(fil_camer)
        plt.title('media filter')

        plt.subplot(2, 2, 3)
        plt.imshow(media_camer)
        plt.title('mean filter')

        plt.show()

def gauss_filter():
    gen = get_single_img_label()
    for img,label in gen:
        resize_img = resize_pic(img,[96,96])
        gauss_img = filters.gaussian(resize_img,sigma=0.9)

        medium_img = rank.median(resize_img,selem=disk(3))



        plt.figure()
        plt.subplot(2,2,1)
        plt.title('ori 28x28')
        plt.imshow(img)

        plt.subplot(2,2,2)
        plt.imshow(resize_img)
        plt.title('resize 96x96')

        plt.subplot(2, 2, 3)
        plt.imshow(gauss_img)
        plt.title('gauss filter')

        # plt.subplot(2, 2, 4)
        # plt.imshow(medium_img)
        plt.show()

def strage_1():
    gen = get_single_img_label()
    for img, label in gen:
        resize_img = resize_pic(img, [96, 96])

        medium_img = rank.median(resize_img, selem=disk(5))
        otsu_img = filters.threshold_otsu(medium_img)
        otsu_img = (medium_img > otsu_img)*1
        otsu_img = np.asanyarray(otsu_img, dtype=np.float32)


        plt.figure()
        plt.subplot(2,2,1)
        plt.title(np.argmax(label))
        plt.imshow(img)

        plt.subplot(2,2,2)
        plt.imshow(resize_img)

        plt.subplot(2, 2, 3)
        plt.imshow(medium_img)

        plt.subplot(2, 2, 4)
        plt.imshow(otsu_img)
        plt.show()



def main():
    # gen_salt_filter()
    gauss_filter()
    # strage_1()

if __name__ == "__main__":
    main()
