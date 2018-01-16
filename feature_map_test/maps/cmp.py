#!/usr/bin/env python3
import argparse
import sys, os
import tempfile

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc

FLAGS = None

dir1 = None
dir2 = None


def load_data():
    global dir1, dir2
    dir1 = os.listdir('./scn1_conv1/')
    dir2 = os.listdir('./scn2_conv1/')
    pass


def get_img(path):
    x = mpimg.imread(path)
    # x = scipy.misc.imresize(x, [process_size[0], process_size[1], 3])
    # x = np.stack([x, x, x], axis=2)
    return x


def show_img(imgs):
    fig = plt.figure(figsize=(5, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(imgs[0])
    fig.add_subplot(2, 2, 4)
    plt.imshow(imgs[1])
    plt.show()
    # plt.waitforbuttonpress()
    plt.close()


def cmp(img1, img2):
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    return np.abs(np.sum(img1 - img2))


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


load_data()
for img1_name in dir1:
    num_min = 999999
    min_ = None
    img_min = None
    for img2_name in dir2:
        img1 = get_img('./scn1_conv1/' + img1_name)
        img2 = get_img('./scn2_conv1/' + img2_name)
        print(mse(img1, img2))
        if (mse(img1, img2) < num_min):
            min_ = [img1_name, img2_name]
            num_min = mse(img1, img2)
            img_min = [img1, img2]
        pass
    print(min_)
    show_img(img_min)
    pass
