import numpy as np
import pandas as pd
from PIL import Image
import os
import random

# set these
#car_image_directory = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
#mask_image_directory = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'

car_image_directory = '/home/nes/Desktop/Caravana/input/train/'
mask_image_directory = '/home/nes/Desktop/Caravana/input/train_masks/'

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
    return pixelvals, mask

# fetches the first num_fetch images in the directory
def nonrandom_image_reader(list_of_images, total_num_images, scale_factor, num_fetch):
    chosen_images = list_of_images[0:num_fetch]
    root_names = [chosen_images[i].split('.')[0] for i in range(num_fetch)]
    mask_names = [root_names[i] + '_mask.gif' for i in range(num_fetch)]

if __name__ == '__main__':

    a, _, c = get_images_masks(car_image_directory, mask_image_directory)

    img, mask = random_image_reader(a, int(c), scale_factor=1.0)
    print(np.shape(img))











