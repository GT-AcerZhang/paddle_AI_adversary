# coding=utf-8

# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
import paddle as paddle
import cv2
import shutil
import math

from models.resnext import ResNeXt50_32x4d
from models.mobilenet_v2 import MobileNetV2_x2_0
from models.inception_v4 import InceptionV4
from models.resnet import ResNet50
from models.vgg import VGG19
from models.densenet import DenseNet121
from models.dpn import DPN68
from models.xception import Xception71, Xception65, Xception41
from models.googlenet import GoogLeNet
from models.resnet import ResNet152
from models.darknet import DarkNet53
from models.efficientnet import EfficientNetB0, EfficientNetB3, EfficientNetB7
from utils import calc_mse, save_adv_image, tensor2img, init_prog, process_img
from PIL import Image
from score import call_avg_mse_np

# 通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
# 比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'
pretrained_models = [
    {'model': ResNeXt50_32x4d, "path": "./pretrained/ResNeXt50_32x4d", "prefix": "resxnet_", 'enable': True},
    {'model': MobileNetV2_x2_0, "path": './pretrained/MobileNetV2', "prefix": "mobile_", 'enable': True},
    {'model': InceptionV4, "path": './pretrained/InceptionV4_pretrained_gau_1.0', "prefix": "", 'enable': True},
    {'model': VGG19, "path": './pretrained/VGG19_pretrained', "prefix": "vgg_", 'enable': True},
    {'model': ResNeXt50_32x4d, "path": './pretrained/ResNeXt50_AdvRetrain_0828', "prefix": "adv_", 'enable': True},
    {'model': Xception71, "path": './pretrained/Xception71_pretrained_gau_1.0', "prefix": "xce71_", 'enable': True},
    {'model': Xception65, "path": './pretrained/Xception65_pretrained_gau_1.0', "prefix": "xce65_", 'enable': True},
    {'model': Xception41, "path": './pretrained/Xception41_pretrained_gau_1.0', "prefix": "xce41_", 'enable': True},
    {'model': GoogLeNet, "path": './pretrained/GoogleNet_pretrained_gau_1.0', "prefix": "google_", 'enable': True},
    {'model': DPN68, "path": './pretrained/DPN68_pretrained_gau_1.0', "prefix": "dpn68_", 'enable': False},
    {'model': DarkNet53, "path": './pretrained/DarkNet53_pretrained_gau_1.0', "prefix": "darknet53_", 'enable': False},
    {'model': EfficientNetB7, "path": './pretrained/EfficientNetB7_pretrained_gau_1.0', "prefix": "effb7_",
     'enable': False},
    {'model': EfficientNetB3, "path": './pretrained/EfficientNetB3_pretrained_gau_1.0', "prefix": "effb3_",
     'enable': True},
    {'model': EfficientNetB0, "path": './pretrained/EfficientNetB0_pretrained_gau_1.0', "prefix": "effb0_",
     'enable': True},
    {'model': DenseNet121, "path": './pretrained/DenseNet121_pretrained', "prefix": "dense_", 'enable': False},
    {'model': ResNet152, "path": './pretrained/ResNet152_pretrained_adv_1.0', "prefix": "resnet152_", 'enable': False},
]

batch_size = 3


def test_set():
    test_path = 'datasets/input_image/val_list.txt'
    with open(test_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    # np.random.shuffle(lines)
    def reader():
        for line in lines:
            label, filename = line.split()
            img = process_img(os.path.join('datasets/input_image/', filename))
            yield img, int(label), filename

    return reader


def load_params(exe):
    for pretrained_model in pretrained_models:
        if pretrained_model['enable']:
            path = pretrained_model['path']
            prefix = pretrained_model['prefix']
            if prefix is not "":
                if os.path.exists(path):
                    param_files = os.listdir(path)
                    for param_file in param_files:
                        if prefix not in param_file:
                            shutil.move(os.path.join(path, param_file),
                                        os.path.join(path, prefix + param_file))
            path_name = path.split('/')[-1]
            parts = path_name.split('_')
            if len(parts) > 1:
                path_name = parts[0] + '_' + parts[1]
            else:
                path_name = parts[0]
            if not os.path.exists('./image_net_pretrained/' + path_name):
                file_count = len(os.listdir(path)) - 3
            else:
                if prefix == 'google_':
                    file_count = len(os.listdir('./image_net_pretrained/' + path_name)) + 10
                else:
                    file_count = len(os.listdir('./image_net_pretrained/' + path_name)) + 2
            count = 0

            def if_exist(var):
                nonlocal count
                res = os.path.exists(os.path.join(path, var.name))
                if res:
                    count = count + 1
                return res

            fluid.io.load_vars(exe, path, main_program=fluid.default_main_program(),
                               predicate=if_exist)
            print("load from {} : {} : {}".format(path, file_count, count))


def create_net(input, label):
    outs = []
    out2 = out3 = 0.0
    include_google = False
    for i, pretrained_model in enumerate(pretrained_models):
        if not with_gpu and i > 0:
            pretrained_model['enable'] = False
        prefix = pretrained_model['prefix']
        enable = pretrained_model['enable']
        if enable:
            if prefix != '':
                model = pretrained_model['model'](prefix=prefix)
            else:
                model = pretrained_model['model']()
            if prefix == 'google_':
                out1, out2, out3 = model.net(input, class_dim=121)
                out = out1
                include_google = True
            else:
                logits = model.net(input, class_dim=121)
                out = fluid.layers.softmax(logits)

            outs.append(out)

    one_hot = fluid.layers.one_hot(label, depth=121)
    out = 0
    for o in outs:
        out = out + o
    if include_google:
        out = out + 0.3 * out2 + 0.3 * out3
        out = out / (len(outs) + 0.6)
    else:
        out = out / len(outs)

    loss = out * one_hot
    loss = fluid.layers.reduce_mean(loss,dim=0)
    loss = fluid.layers.reduce_sum(loss)
    loss = fluid.layers.log(1 - loss)
    loss = 0 - loss
    loss = fluid.layers.reduce_mean(loss)

    return loss, outs


def main(use_cuda):
    """
    Advbox demo which demonstrate how to use advbox.
    """
    main_prog = fluid.default_main_program()
    output_target = './datasets/output_image/'
    if not os.path.exists(output_target):
        os.makedirs(output_target)
    IMG_NAME = 'img'
    LABEL_NAME = 'label'
    global_id = 0

    img = fluid.layers.data(name=IMG_NAME, shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
    noise = fluid.layers.create_parameter(name="noise", shape=[batch_size, 3, 224, 224], dtype='float32',
                                          default_initializer=fluid.initializer.Constant(0.0000001))

    true_image = noise + img
    
    r_image = fluid.layers.crop(true_image, shape=[batch_size, 1, 224, 224], offsets=[0, 0, 0, 0], name='r_image')
    g_image = fluid.layers.crop(true_image, shape=[batch_size, 1, 224, 224], offsets=[0, 1, 0, 0], name='g_image')
    b_image = fluid.layers.crop(true_image, shape=[batch_size, 1, 224, 224], offsets=[0, 2, 0, 0], name='b_image')

    
    max_mean = [0.485, 0.456, 0.406]
    max_std = [0.229, 0.224, 0.225]
    r_max = (1 - max_mean[0]) / max_std[0]
    g_max = (1 - max_mean[1]) / max_std[1]
    b_max = (1 - max_mean[2]) / max_std[2]

    r_min = (0 - max_mean[0]) / max_std[0]
    g_min = (0 - max_mean[1]) / max_std[1]
    b_min = (0 - max_mean[2]) / max_std[2]
    r_image = fluid.layers.clip(x=r_image, min=r_min, max=r_max)
    g_image = fluid.layers.clip(x=g_image, min=g_min, max=g_max)
    b_image = fluid.layers.clip(x=b_image, min=b_min, max=b_max)

    true_image = fluid.layers.concat([r_image, g_image, b_image], axis=1)

    loss, outs = create_net(true_image, label)

    std = fluid.layers.assign(np.array([[[0.229]], [[0.224]], [[0.225]]]).astype('float32'))

    square = fluid.layers.square(noise * std * 255.0)
    # avg l2 norm
    loss2 = fluid.layers.reduce_sum(square, dim=1)
    loss2 = fluid.layers.sqrt(loss2)
    loss2 = fluid.layers.reduce_mean(loss2)

    #avg mse
    # loss2 = fluid.layers.reduce_mean(square)

    loss = loss + 0.005 * loss2

    init_prog(main_prog)
    test_prog = main_prog.clone()
    lr = fluid.layers.create_global_var(shape=[1], value=0.02, dtype='float32', persistable=True,
                                        name='learning_rate_0')

    opt = fluid.optimizer.Adam(learning_rate=lr)
    opt.minimize(loss, parameter_list=[noise.name])

    # 根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    test_reader = paddle.batch(test_set(), batch_size=batch_size)

    load_params(exe)

    fail_count = 0

    for block in main_prog.blocks:
        for var in block.vars.keys():
            if 'learning_rate' in var:
                pd_lr = fluid.global_scope().find_var(var)
                print(var)
            if 'beta1_pow_acc' in var:
                pd_noise_beta1 = fluid.global_scope().find_var(var)
                print(var)
            if 'moment1' in var:
                pd_noise_mom1 = fluid.global_scope().find_var(var)
                print(var)
            if 'beta2_pow_acc' in var:
                pd_noise_beta2 = fluid.global_scope().find_var(var)
                print(var)
            if 'moment2' in var:
                pd_noise_mom2 = fluid.global_scope().find_var(var)
                print(var)
    print(np.array(pd_lr.get_tensor()))
    for train_id, data in enumerate(test_reader()):
        images = []
        labels = []
        filenames = []
        for i in range(batch_size):
            images.append(data[i][0][0])
            labels.append([data[i][1]])
            filenames.append(data[i][2])
            # image = data[0][0]
            # label = data[0][1]
            # label = np.array([[label]])
            # filename = data[0][2]
        images = np.array(images)
        labels = np.array(labels)
        for block in main_prog.blocks:
            for param in block.all_parameters():
                if param.name == 'noise':
                    pd_var = fluid.global_scope().find_var(param.name)
                    pd_param = pd_var.get_tensor()
                    print("load: {}, shape: {}".format(param.name, param.shape))
                    print("Before setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
                    noise_tensor = np.zeros(param.shape).astype('float32')
                    noise_tensor[:] = 1e-7
                    pd_param.set(noise_tensor, place)
                    print("After setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
        # pd_lr.get_tensor().set(np.array([0.02]).astype('float32'), place)
        if batch_size > 1:
            pd_noise_beta1.get_tensor().set(np.array([0.9]).astype('float32'), place)
            pd_noise_beta2.get_tensor().set(np.array([0.999]).astype('float32'), place)
            pd_noise_mom1.get_tensor().set(np.zeros(shape=[batch_size, 3, 224, 224]).astype('float32'),place)
            pd_noise_mom2.get_tensor().set(np.zeros(shape=[batch_size, 3, 224, 224]).astype('float32'), place)

        i = 0
        fetch_list = [true_image, lr, loss, loss2, noise]
        mean_np = np.array([[[[0.485]], [[0.456]], [[0.406]]]]).astype('float32')
        std_np = np.array([[[[0.229]], [[0.224]], [[0.225]]]]).astype('float32')
        ori_img = np.round((images * std_np + mean_np) * 255.0)
        ori_img = np.clip(ori_img, 0, 255).astype('uint8')
        while True:
            if i == 0:
                test_vars = exe.run(program=test_prog,
                                    feed={'img': images, 'label': labels},
                                    fetch_list=outs)

                for m in range(batch_size):
                    str = 'First step test network,id:{},'.format(global_id + 1)
                    global_id += 1
                    adv_labels = []
                    for j in range(len(outs)):
                        o = test_vars[j][m]
                        adv_label = np.argmax(o)
                        adv_labels.append(adv_label)
                        str += 'adv{}:%d,'.format(j + 1)
                    print(str % (*adv_labels,))

            train_vars = exe.run(program=fluid.default_main_program(),
                                 feed={'img': images, 'label': labels},
                                 fetch_list=fetch_list)
            n = train_vars[-1]
            l2 = train_vars[-2]
            l1 = train_vars[-3]
            lr1 = train_vars[-4]
            tr_img = train_vars[-5]

            adv_img = n + images
            adv_img = np.round((adv_img * std_np + mean_np) * 255.0)
            adv_img = np.clip(adv_img, 0, 255).astype('uint8')

            diff = adv_img.astype('float32') - ori_img.astype('float32')
            avg_mse = diff * diff
            # avg l2 norm
            l2_norm = np.sum(avg_mse, axis=1)
            l2_norm = np.sqrt(l2_norm)
            l2_norm = np.mean(l2_norm, axis=(1,2))
            # avg mse
            avg_mse = np.mean(avg_mse, axis=(1, 2, 3))


            test_vars = exe.run(program=test_prog,
                                feed={'img': images, 'label': labels},
                                fetch_list=outs)
            successful = batch_size * len(outs)
            for m in range(batch_size):
                str = 'batch:%d,id:{},lr:%f,loss1:%f,loss2:%f,avg_mse:%f,l2_norm:%f,'.format(train_id * batch_size + m + 1)
                adv_labels = []
                for j in range(len(outs)):
                    o = test_vars[j][m]
                    adv_label = np.argmax(o)
                    adv_labels.append(adv_label)
                    str += 'adv{}:%d,'.format(j + 1)
                print(str % (i, lr1, l1, l2, avg_mse[m], l2_norm[m], *adv_labels))

                for adv_label in adv_labels:
                    if adv_label == labels[m]:
                        successful -= 1

            i += 1
            if (successful >= batch_size * len(outs) - 1 and np.mean(l2_norm) < 1.0) or i == 3000:
                if successful >= batch_size * len(outs) - 1:
                    print('attack successful')
                else:
                    print('attack failed')
                    fail_count += 1
                break

        print("failed:%d" % (fail_count,))

        adv_img = adv_img.astype('float32') / 255.0
        adv_img = adv_img - mean_np
        adv_img = adv_img / std_np

        for m in range(batch_size):
            adv_image = tensor2img(adv_img[m][np.newaxis,:,:,:])
            ori_image = tensor2img(images[m][np.newaxis,:,:,:])

            print('id:{},mse:{}'.format(train_id * batch_size + m + 1 ,call_avg_mse_np(adv_image, ori_image)))
            save_adv_image(adv_image, os.path.join(output_target, filenames[m].split('.')[0] + '.png'))
    print("attack over ,failed:%d" % (fail_count,))


if __name__ == '__main__':
    main(use_cuda=with_gpu)
