import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage import util
from skimage.morphology import disk
from skimage.filters import rank
from matplotlib import pyplot as plt
from skimage import filters
from skimage import feature
from skimage import morphology

from mnistCompRes.mnisttool import imgUtil

from PIL import Image

mnist = input_data.read_data_sets("../MNIST_DATA/", one_hot=True)

def get_single_img_label():
    '''
    获得单张图片和对应的label，图片已经被转化为2D形状，label转化为对应的数字
    '''
    for epoch in range(10000):
        images, labels = mnist.train.next_batch(10,shuffle=True)
        for it in range(10):
            img = np.reshape(images[it], newshape=(28, 28))
            yield (img,np.argmax(labels[it]))

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
    '''
    对图片进行放大，然后进行高斯滤波，然后进行中值滤波
    '''
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

def canny_filter():
    '''
    对高斯滤波后的图像进行canny滤波
    '''
    gen = get_single_img_label()
    for img, label in gen:
        img = resize_pic(img, [96, 96])
        gauss_img = filters.gaussian(img, sigma=0.9)
        medium_img = rank.median(gauss_img,selem=disk(3))


        canny_img = feature.canny(gauss_img, sigma=3)*1
        canny_img2 = feature.canny(medium_img,sigma=3)*1

        canny_img = np.asanyarray(canny_img,dtype=np.float32)
        canny_img2 = np.asanyarray(canny_img2,dtype=np.float32)

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title(label)
        plt.imshow(canny_img2)

        plt.subplot(2, 2, 2)
        plt.imshow(gauss_img)
        plt.title('gauss_img')

        plt.subplot(2, 2, 3)
        plt.imshow(medium_img)
        plt.title('medium_img')

        plt.subplot(2, 2, 4)
        plt.imshow(canny_img)
        plt.title('canny_img')
        plt.show()

def skeleton_retrive():
    '''
    提取数字图片的骨架
    '''
    gen = get_single_img_label()
    for img, label in gen:
        img = resize_pic(img, [96, 96])
        gauss_img = filters.gaussian(img, sigma=0.9)
        medium_img = rank.median(gauss_img, selem=disk(3))

        val = filters.threshold_otsu(medium_img)
        twovalue = (medium_img > val)*1
        twovalue = np.asanyarray(twovalue,dtype=np.uint8)

        skeleton_img = morphology.skeletonize(twovalue)


        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(gauss_img)

        plt.subplot(1, 2, 2)
        plt.imshow(skeleton_img)
        plt.show()

def same_number():
    '''
    相同数字的图片
    '''
    gen = get_single_img_label()
    for img, label in gen:
        if(label == 6):
            plt.figure()
            plt.imshow(img)
            plt.show()

def gabor_filter():
    gen = get_single_img_label()
    for img, label in gen:
        img = resize_pic(img, [96, 96])
        gauss_img = filters.gaussian(img, sigma=0.9)
        gabor_real0,gabor_virt0 = filters.gabor(gauss_img,frequency=0.8, theta=0)
        gabor_real45,gabor_virt45 = filters.gabor(gauss_img,frequency=0.8, theta=45)
        gabor_real90,gabor_virt90 = filters.gabor(gauss_img,frequency=0.9, theta=90)
        gabor_real135,gabor_virt135 = filters.gabor(gauss_img,frequency=0.9, theta=135)

        plt.figure()
        plt.subplot(2,3,1)
        plt.imshow(img)
        plt.title('ori')

        plt.subplot(2, 3, 2)
        plt.imshow(gauss_img)
        plt.title('gauss')

        plt.subplot(2, 3, 3)
        plt.imshow(gabor_real0)
        plt.title('0')

        plt.subplot(2, 3, 4)
        plt.imshow(gabor_real45)
        plt.title('45')

        plt.subplot(2, 3, 5)
        plt.imshow(gabor_real90)
        plt.title('90')

        plt.subplot(2, 3, 6)
        plt.imshow(gabor_real135)
        plt.title('135')

        plt.show()

def activate_function():
    plt.figure()

    ax = plt.subplot(3,2,1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data',0))
    ax.set_xlim(-6,6)
    ax.set_ylim(-1,1)
    x = np.linspace(-5,5,1000)
    y = [1/(1+np.exp(-i)) for i in x]
    plt.title('Sigmoid')
    plt.plot(x,y)

    ax = plt.subplot(3,2,2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 1)
    x = np.linspace(-5,5,1000)
    y = [(np.exp(i)-np.exp(-i))/(np.exp(i)+np.exp(-i)) for i in x]
    plt.title('Tanh')
    plt.plot(x,y)

    ax = plt.subplot(3,2, 3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 1)
    x = np.linspace(-5, 5, 1000)
    y = []
    for i in x:
        if i<0 :
            y.append(0)
        else:
            y.append(i)
    plt.title('Relu')
    plt.plot(x, y)

    ax = plt.subplot(3, 2, 4)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 1)
    x = np.linspace(-5, 5, 1000)
    y = []
    for i in x:
        if i < 0:
            y.append(0.1*i)
        else:
            y.append(i)
    plt.title('LeakyRelu')
    plt.plot(x, y)

    ax = plt.subplot(3, 2, 5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 1)
    x = np.linspace(-5, 5, 1000)
    y = []
    for i in x:
        if i < 0:
            y.append(0.1 * (np.exp(i) - 1))
        else:
            y.append(i)
    plt.title('ELU')
    plt.plot(x, y)




    plt.show()

def aserialPic():
    gen = get_single_img_label()
    already = []
    pic = []
    for img, label in gen:
        if label not in already:
            already.append(label)
            pic.append(img)
            if len(pic) == 10:
                break

    plt.figure()

    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.imshow(pic[i])
    plt.show()

def img_aug():
    gen = get_single_img_label()
    for img, label in gen:
        resize_img = resize_pic(img, [96, 96])

        while True:
        # for i in range(10):
            obj = imgUtil.single_image_aug_gray(resize_img)
            plt.figure()
            plt.imshow(obj)
            # plt.title(i)
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
        plt.title(label)
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
    # gauss_filter()
    # strage_1()
    # canny_filter()
    # skeleton_retrive()
    # same_number()
    # gabor_filter()
    # activate_function()
    # aserialPic()
    img_aug()

if __name__ == "__main__":
    main()
