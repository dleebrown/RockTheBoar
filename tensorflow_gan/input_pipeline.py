from __future__ import print_function
import numpy as np
import pandas as pd
from PIL import Image
import os
import random

# set these
car_image_directory = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
mask_image_directory = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'

#car_image_directory = '/home/nes/Desktop/Caravana/input/train/'
#mask_image_directory = '/home/nes/Desktop/Caravana/input/train_masks/'

def downsample_img(input_img, scale_factor):
    img_size = input_img.size
    input_img = input_img.resize((int(img_size[0]*scale_factor), int(img_size[1]*scale_factor)))
    return input_img


def normalize_img(np_array, normalization):
    np_array = np_array/normalization
    return np_array


def image_input(car_image_title, mask_image_title, scale_factor):
    mask_img = Image.open(mask_image_title)
    mask_img = downsample_img(mask_img, scale_factor)
    mask_img = np.array(mask_img, dtype=np.float32)
    # mask_img = normalize_img(mask_img, 255.)

    car_img = Image.open(car_image_title)
    car_img = downsample_img(car_img, scale_factor)
    car_img = np.array(car_img, dtype=np.float32)
    car_img = normalize_img(car_img, 255.)
    return car_img, mask_img

# This function is to be used to run the infer code on the test data

def no_mask_image_input(car_image_title, scale_factor):
    # mask_img = Image.open(mask_image_title)
    # mask_img = downsample_img(mask_img, scale_factor)
    # mask_img = np.array(mask_img, dtype=np.float32)
    # mask_img = normalize_img(mask_img, 255.)

    car_img = Image.open(car_image_title)
    car_img = downsample_img(car_img, scale_factor)
    car_img = np.array(car_img, dtype=np.float32)
    car_img = normalize_img(car_img, 255.)
    return car_img


def get_images_masks(rgb_dir, mask_dir):
    image_list = [img for img in os.listdir(rgb_dir) if '.jpg' in img]
    mask_list = [masks for masks in os.listdir(mask_dir) if '.gif' in masks]
    num_images = len(image_list)
    return image_list, mask_list, num_images


def random_image_reader(list_of_images, total_num_images, scale_factor):
    chosen_image = random.randint(0, total_num_images)
    chosen_image = list_of_images[chosen_image]
    root_name = chosen_image.split('.')[0]
    mask_name = root_name + '_mask.gif'
    pixelvals, mask = image_input(car_image_directory + chosen_image, mask_image_directory + mask_name, scale_factor)
    # print(root_name+'.jpg, ', end='') # added to match submission format'
    #print(root_name+'.jpg, ',end="",flush=True) # added to match submission format'
    return pixelvals, mask

# non-random image reader - basically the same as above but takes in an index of the image to read
def not_random_image_reader(list_of_images, total_num_images, scale_factor, whichimage):
    chosen_image = whichimage
    chosen_image = list_of_images[chosen_image]
    root_name = chosen_image.split('.')[0]
    mask_name = root_name + '_mask.gif'
    pixelvals = no_mask_image_input(car_image_directory + chosen_image, scale_factor)
    im_name = root_name+'.jpg'
    return pixelvals, im_name

if __name__ == '__main__':

    a, _, c = get_images_masks(car_image_directory, mask_image_directory)

    img, mask = random_image_reader(a, int(c), scale_factor=1.0)
    print(np.shape(img))











