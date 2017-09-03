import numpy as np
import pandas as pd
from PIL import Image
import os
import random

car_image_directory = '__RGB_CAR_IMAGES_DIRECTORY__'
mask_image_directory = '__B&W_MASK_IMAGES_DIRECTORY__'

def image_input(mask_image_title, car_image_title):

    mask_img = Image.open(mask_image_title)
    mask_img = np.array(mask_img)

    car_img = Image.open(car_image_title)
    car_img = np.array(car_img)

    return car_img, mask_img

def get_images_masks(rgb_dir, mask_dir):
    image_list = [img for img in os.listdir(rgb_dir) if '.jpg' in img]
    mask_list = [masks for masks in os.listdir(mask_dir) if '.gif' in masks]
    num_images = len(image_list)
    return image_list, mask_list, num_images

def random_image_reader(list_of_images, total_num_images):
    chosen_image = random.randint(0, total_num_images)
    chosen_image = list_of_images[chosen_image]
    root_name = chosen_image.split('.')[0]
    mask_name = root_name + '_mask.gif'
    pixelvals, mask = image_input(car_image_directory + chosen_image, mask_image_directory + mask_name)
    return pixelvals, mask

a, _, c = get_images_masks(car_image_directory, mask_image_directory)

print(random_image_reader(a, int(c)))











