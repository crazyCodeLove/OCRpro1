#coding=utf-8
# from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
from skimage import data
from skimage.filters import rank
from skimage import io
from skimage.transform import rotate
from skimage.transform import AffineTransform,warp
from skimage.transform import rescale
from skimage.transform import resize
from skimage import color
from skimage.morphology import disk
from skimage import util
from matplotlib import pyplot as plt
from skimage import filters


def batch_imgs_resize(batch_imgs, des_img_size, hps):
    """

    :param batch_imgs: images shape must be a 4D tensor,
    :param des_img_size: a single integer, des image size(width,height)
    :param hps:
    :return:
    """
    oldtype = batch_imgs.dtype

    result = np.zeros([hps.batch_nums, des_img_size, des_img_size, hps.img_depth], dtype=oldtype)
    for it in range(hps.batch_nums):
        img = batch_imgs[it]
        img = single_img_resize(img, hps.img_width, des_img_size)
        result[it] = img
    return result

def single_img_resize(img, img_size, des_img_size, pad_width=6):
    """

    :param img:must be a tensor, 3D, [img_width, img_width, img_depth]
    :param img_size: ori img size
    :param img_depth: ori img depth
    :param des_img_size:
    :return:
    """
    ori_img_type = img.dtype
    box_size = img_size + pad_width
    img_depth = img.shape[2]
    if img_depth == 1:
        box = np.zeros([box_size,box_size], dtype=ori_img_type)
        img = np.reshape(img,[img_size, img_size])
    else:
        box = np.zeros([box_size,box_size,img_depth], dtype=ori_img_type)

    box = Image.fromarray(box)
    img = Image.fromarray(img)
    box_edge = [(pad_width//2),(pad_width//2),(pad_width//2)+img_size,(pad_width//2)+img_size]
    box.paste(img, box_edge)

    box = box.resize([des_img_size,des_img_size])

    img = np.asarray(box)
    img = np.reshape(img, [des_img_size,des_img_size,img_depth])
    return img


def batch_imgs_pre_process(batch_imgs):
    oldtype = batch_imgs.dtype
    tsp = batch_imgs.shape
    new_shape = (tsp[0],tsp[1],tsp[2],4)
    result = np.zeros(new_shape, dtype=oldtype)

    for it in range(batch_imgs.shape[0]):
        img = batch_imgs[it]
        img = single_img_pre_process(img)
        result[it] = img

    return result


def single_img_pre_process(img):
    ori_shape = img.shape
    ori_dtype = img.dtype

    img = np.reshape(img, newshape=(ori_shape[0], ori_shape[1]))
    gab = []
    for i in range(4):
        gabor_real, gabor_virt = filters.gabor(img, frequency=0.8, theta=i*45)
        gab.append(gabor_real)

    result = np.stack(gab,axis=2)
    result = result.astype(dtype=ori_dtype)
    return result

def batch_imgs_aug(batch_imgs, is_arrange_0_to_1 = True):
    """

    :param batch_imgs: images shape is 4D tensor, value is arrange(0,1.0),或者(0,255)
    :param is_arrange_0_to_1: batchImgs的直是否是(0,1.0).
    :return:
    """

    oldtype = batch_imgs.dtype

    result = np.zeros(batch_imgs.shape, dtype=oldtype)

    for it in range(batch_imgs.shape[0]):
        img = batch_imgs[it]
        img = single_image_aug_gray(img)
        result[it] = img

    return result



    #convert imgs from (0,255) --> (0,1.0)
    # images_aug = images_aug * 1.0/255.0

    return images_aug

def batch_imgs_reshape(batch_imgs, hps):
    """
    :param batch_imgs: 2D tensor imgs shape is [batch_nums, hps.img_width*hps.img_width*hps.img_depth]
                                ====>[batch_nums, hps.img_width, hps.img_width, hps.img_depth]
    :param hps:
    :return:
    """
    batch_imgs = np.reshape(batch_imgs, [-1, hps.img_width, hps.img_width, hps.img_depth])
    return batch_imgs

def single_image_aug_gray(img):
    # crop_width=((top,left),(bottom,right))
    ori_shape = img.shape
    ori_dtype = img.dtype

    if(len(img.shape) == 3):
        if(img.shape[2] > 1):
            img = color.rgb2gray(img)
        else:
            img = np.reshape(img,newshape=(ori_shape[0],ori_shape[1]))

    # random crop
    crop_range = (0, 4)

    top = np.random.randint(*crop_range)
    left = np.random.randint(*crop_range)
    bottom = np.random.randint(*crop_range)
    right = np.random.randint(*crop_range)
    result = util.crop(img,crop_width=((top,left),(bottom,right)))

    # random scale
    scale_range = (0.7, 1.2)
    scale_x = np.random.uniform(*scale_range)
    scale_y = np.random.uniform(*scale_range)
    result = rescale(result,scale=(scale_x,scale_y))

    # restore orignal size
    result = resize(result,output_shape=(ori_shape[0],ori_shape[1]))

    # random rotate
    rotate_range = (-20,20)
    rotate_angle = np.random.randint(*rotate_range)
    result = rotate(result,angle=rotate_angle)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.title("ori")
    # plt.imshow(img)
    #
    # plt.subplot(1, 2, 2)
    # plt.title("result")
    # plt.imshow(result)
    # plt.show()

    result = np.reshape(result,newshape=ori_shape)
    result = result.astype(dtype=ori_dtype)
    return result

def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def main():
    camera = data.camera()
    # rescale_camera = rescale(camera, scale=0.1)
    # resize_camera = resize(camera,output_shape=(500,500))
    # newcamera = rotate(camera,15)
    # crop_width=((top,left),(bottom,right))
    # crop_right_camera = util.crop(camera,crop_width=((0,0),(0,150)))
    # crop_left_camera = util.crop(camera,crop_width=((0,150),(0,0)))
    # crop_center_camera = util.crop(camera,crop_width=((150,150),(150,150)))
    # tform = AffineTransform(scale=(1.1,1.1),shear=0.7)
    # sh_ro7 = warp(camera,inverse_map=tform,output_shape=(512,512))
    #
    # tform = AffineTransform(shear=0.8, rotation=90)
    # sh_ro8 = warp(camera, inverse_map=tform, output_shape=(512, 512))
    tform = AffineTransform(shear=-0.2,translation=(50,0))
    sh_ro9 = warp(camera, inverse_map=tform.inverse, output_shape=(512, 512))

    # camera = util.random_noise(camera,mode='salt')


    # fil_camer = rank.median(camera,selem=disk(3))
    # media_camer = rank.mean(camera,selem=disk(3))



    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(camera)
    plt.subplot(2, 2, 3)
    plt.imshow(sh_ro9)

    plt.show()


if __name__ == "__main__":
    main()
