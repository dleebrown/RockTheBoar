import numpy as np
from PIL import Image
import os


"""Just a test script to look at the data while I build the NN
"""

data_loc = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
files = [img for img in os.listdir(data_loc) if '.jpg' in img]

img_dims = np.zeros((len(files), 2))

counter = 0
for filename in files:
    with Image.open(data_loc+filename) as current_image:
        img_dims[counter, 0], img_dims[counter, 1] = current_image.size
    counter += 1

print('min/max width: '+str(np.min(img_dims[:, 0]))+'/'+str(np.max(img_dims[:, 0])))
print('min/max height: '+str(np.min(img_dims[:, 1]))+'/'+str(np.max(img_dims[:, 1])))


