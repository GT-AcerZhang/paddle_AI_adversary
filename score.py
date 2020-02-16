import glob
import os
import cv2
import numpy as np

def call_avg_mse_np(input1, input2):
    output_img = np.array(input1).astype('float32')
    input_img = np.array(input2).astype('float32')
    diff = output_img - input_img
    diff = diff * diff
    # avg l2 norm
    # sum = np.sum(diff, axis=2)
    # sum = np.sqrt(sum)
    # mean = np.mean(sum)
    #avg mse
    mean = np.mean(diff)
    return mean

def call_avg_mse(input1, input2):
    output_img = cv2.imread(input1)
    output_img = cv2.resize(output_img, (224, 224))
    input_img = cv2.imread(input2)
    input_img = cv2.resize(input_img, (224, 224))
    output_img = np.array(output_img).astype('float32')
    input_img = np.array(input_img).astype('float32')
    diff = output_img - input_img
    diff = diff * diff
    sum = np.sum(diff, axis=2)
    sum = np.sqrt(sum)
    mean = np.mean(sum)
    return mean

if __name__ == '__main__':
    input_path = 'datasets/input_image'
    output_path = 'datasets/output_image'
    inputs = glob.glob(os.path.join(input_path, '*.jpg'))
    total_mean = 0.0
    for input_file in inputs:
        input_name = input_file.split('/')[-1].split('.')[0]
        output_file = os.path.join(output_path, input_name + '.png')
        if os.path.exists(output_file):
            mean = call_avg_mse(input_file, output_file)
            print(mean)
            total_mean += mean
    print('total mean:{}'.format(total_mean / 120.0))
