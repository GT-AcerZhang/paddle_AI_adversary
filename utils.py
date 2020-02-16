# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import time
import sys
import math
import numpy as np
import argparse
import functools
import distutils.util
import six

from PIL import Image, ImageOps
# 绘图函数
import matplotlib
# 服务器环境设置
import matplotlib.pyplot as plt
from image_proccess import apply_affine_transform
import random

# 去除batch_norm的影响
def init_prog(prog):
    for op in prog.block(0).ops:
        # print("op type is {}".format(op.type))
        if op.type in ["batch_norm"]:
            # 兼容旧版本 paddle
            if hasattr(op, 'set_attr'):
                op.set_attr('is_test', False)
                op.set_attr('use_global_stats', True)
            else:
                op._set_attr('is_test', False)
                op._set_attr('use_global_stats', True)
                op.desc.check_attrs()
        if op.type in ["dropout"]:
            if hasattr(op, 'set_attr'):
                op.set_attr('dropout_prob', 0.0)
            else:
                op._set_attr('dropout_prob',0.0)


def img2tensor(img, image_shape=[3, 224, 224]):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = cv2.resize(img, (image_shape[1], image_shape[2]))

    # RGB img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255

    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)

    return img


def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def img_diversity(imgs):
    res = []
    for img in imgs:
        img = img.transpose([1,2,0])
        img = Image.fromarray(img, mode='RGB')
        r1 = random.random()
        if r1 > 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # # 随机垂直翻转
        r2 = random.random()
        if r2 > 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # # 随机角度翻转
        # r3 = random.randint(-30, 30)
        # img = img.rotate(r3, expand=False)
        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)

        row_axis_index = 0
        col_axis_index = 1
        channel_axis_index = 2
        img_shape = img.shape

        rotation_range = 15
        theta = np.random.uniform(-rotation_range, rotation_range)

        height_shift_range = 0.1
        tx_img = np.random.uniform(-height_shift_range,
                                   height_shift_range)
        if np.max(height_shift_range) < 1:
            tx_img *= img_shape[row_axis_index]
        width_shift_range = 0.1
        ty_img = np.random.uniform(-width_shift_range,
                                   width_shift_range)
        if np.max(width_shift_range) < 1:
            ty_img *= img_shape[col_axis_index]
        shear_range = 0.0
        shear = np.random.uniform(-shear_range, shear_range)
        zoom_range = [1 - 0.0, 1 + 0.0]
        zx, zy = np.random.uniform(
            zoom_range[0],
            zoom_range[1],
            2)

        img = apply_affine_transform(img, theta, tx_img, ty_img, shear, zx, zy, row_axis_index, col_axis_index,
                                     channel_axis_index, fill_mode='constant')

        # im = img / 255.0
        ###display
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(im)
        # plt.show()
        ####
        img = img.transpose([2,1,0])
        res.append(img)
    return np.array(res)

def process_img(img_path="", image_shape=[3, 224, 224]):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_shape[1], image_shape[2]))
    # img = cv2.resize(img,(256,256))
    # img = crop_image(img, image_shape[1], True)

    # RBG img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    # img = img.astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)

    return img


def tensor2img_final(tensor):
    img = tensor.copy()
    img = np.clip(img, 0, 255)
    img = img[0].astype(np.uint8)
    img = img.transpose(1, 2, 0)
    img = img[:, :, ::-1]
    return img

def tensor2img(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))

    img = tensor.copy()

    img *= img_std
    img += img_mean

    img = np.round(img * 255)
    img = np.clip(img, 0, 255)

    img = img[0].astype(np.uint8)

    img = img.transpose(1, 2, 0)
    img = img[:, :, ::-1]

    return img


def save_adv_image(img, output_path):
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return


def calc_mse(org_img, adv_img):
    diff = org_img.astype('int32').reshape((-1, 3)) - adv_img.astype('int32').reshape((-1, 3))
    distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
    return distance


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)
import paddle.fluid as fluid
def input_diversity(input_tensor, h_flip=False, v_flip=False, angle=0.0, w_shift=0.0, h_shift=0.0,shear=0.0,zx=0.0,zy=0.0):
    zero = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.0)
    one = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.0)
    if h_flip:
        prob = fluid.layers.uniform_random([1], dtype='float32', min=0.0, max=1.0)
        if fluid.layers.greater_than(prob, fluid.layers.fill_constant(shape=[1],dtype='float32',value=0.5)):
            input_tensor = fluid.layers.reverse(input_tensor,axis=[3])
    if v_flip:
        prob = fluid.layers.uniform_random([1], dtype='float32', min=0.0, max=1.0)
        if fluid.layers.greater_than(prob, fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.5)):
            input_tensor = fluid.layers.reverse(input_tensor, axis=[2])
    #angle
    if angle != 0:
        degree = fluid.layers.uniform_random([1], dtype='float32', min=-angle, max=angle)
        rnd = degree * math.pi / 180.0

        angle_theta = fluid.layers.concat([fluid.layers.cos(rnd), fluid.layers.sin(0 - rnd), zero,
                                        fluid.layers.sin(rnd), fluid.layers.cos(rnd), zero,
                                           zero,zero,one])
        angle_theta = fluid.layers.reshape(angle_theta, shape=[3, 3])
        transform_theta = angle_theta

    #shift
    if w_shift != 0 or h_shift != 0:
        width_shift = fluid.layers.uniform_random(shape=[1],dtype='float32',min=-w_shift, max=w_shift)
        height_shift = fluid.layers.uniform_random(shape=[1],dtype='float32',min=-h_shift, max=h_shift)
        shift_theta = fluid.layers.concat([one, zero, width_shift,
                                     zero, one, height_shift,
                                           zero,zero,one])
        shift_theta = fluid.layers.reshape(shift_theta, shape=[3, 3])
        if transform_theta is None:
            transform_theta = shift_theta
        else:
            transform_theta = fluid.layers.mul(transform_theta, shift_theta)

    if shear != 0:
        shear = fluid.layers.uniform_random([1], dtype='float32', min=-shear, max=shear)
        shear = shear * math.pi / 180.0
        shear_theta = fluid.layers.concat([one, 0 - fluid.layers.sin(shear), zero,
                                           zero, fluid.layers.cos(shear), zero,
                                           zero, zero, one])
        shear_theta = fluid.layers.reshape(shear_theta, shape=[3, 3])
        if transform_theta is None:
            transform_theta = shear_theta
        else:
            transform_theta = fluid.layers.mul(transform_theta, shear_theta)

    if zx != 0 or zy != 0:
        zx = fluid.layers.uniform_random([1], dtype='float32', min=1.0 - zx, max=1.0 + zx)
        zy = fluid.layers.uniform_random([1], dtype='float32', min=1.0 - zy, max=1.0 + zy)
        zoom_theta = fluid.layers.concat([zx, zero, zero,
                                           zero, zy, zero,
                                           zero, zero, one])
        zoom_theta = fluid.layers.reshape(zoom_theta, shape=[3, 3])
        if transform_theta is None:
            transform_theta = zoom_theta
        else:
            transform_theta = fluid.layers.mul(transform_theta, zoom_theta)


    transform_theta = fluid.layers.crop(transform_theta,shape=[2,3],offsets=[0,0])
    transform_theta = fluid.layers.reshape(transform_theta,shape=[1, 2, 3])
    grid = fluid.layers.affine_grid(transform_theta, out_shape=input_tensor.shape)
    input_tensor = fluid.layers.grid_sampler(input_tensor, grid)

    return input_tensor
