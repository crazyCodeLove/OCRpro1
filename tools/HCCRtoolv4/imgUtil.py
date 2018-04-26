#coding=utf-8


from imgaug import augmenters as iaa
import numpy as np
from PIL import Image

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
        img = single_img_resize(img, hps.img_width, hps.img_depth, des_img_size)
        result[it] = img
    return result

def single_img_resize(img, img_size, img_depth, des_img_size, pad_width=6):
    """

    :param img:must be a tensor, 3D, [img_width, img_width, img_depth]
    :param img_size: ori img size
    :param img_depth: ori img depth
    :param des_img_size:
    :return:
    """
    ori_img_type = img.dtype
    box_size = img_size + pad_width
    if img_depth == 1:
        box = np.zeros([box_size,box_size], dtype=ori_img_type)
        img = np.reshape(img,[img_size, img_size])
    else:
        box = np.zeros([box_size,box_size,img_depth], dtype=ori_img_type)

    box = Image.fromarray(box)
    img = Image.fromarray(img)
    box_edge = [(pad_width/2),(pad_width/2),(pad_width/2)+img_size,(pad_width/2)+img_size]
    box.paste(img, box_edge)

    box = box.resize([des_img_size,des_img_size])

    img = np.asarray(box)
    img = np.reshape(img, [des_img_size,des_img_size,img_depth])
    return img


def batch_imgs_aug(batchImgs,is_arrange_0_to_1 = True):
    """

    :param batchImgs: images shape is 4D tensor, value is arrange(0,1.0),或者(0,255)
    :param is_arrange_0_to_1: batchImgs的直是否是(0,1.0).
    :return:
    """
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 2)),
        iaa.Affine(
            scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
            rotate=(-14, 14),
            shear=(-14, 14)
        ),
        # iaa.ElasticTransformation(alpha=38.0, sigma=6.0)
    ])
    if is_arrange_0_to_1:
        batchImgs = np.array((batchImgs * 255), dtype=np.uint8)
    else:
        batchImgs = np.array(batchImgs,dtype=np.uint8)

    images_aug = seq.augment_images(batchImgs)
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