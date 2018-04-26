import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage import util
from skimage.morphology import disk
from skimage.filters import rank
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

def gen_salt_filter():
    images,labels = mnist.train.next_batch(20)
    shape = images.shape
    for it in range(shape[0]):
        img = np.reshape(images[it],newshape=(28,28))
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


def main():
    gen_salt_filter()

if __name__ == "__main__":
    main()