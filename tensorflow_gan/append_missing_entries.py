from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.morphology import opening
from skimage.morphology import disk
import input_pipeline as inpipe
import pandas as pd
import csv

# EXAMPLE FOR USE AT THE BOTTOM OF THIS FILE
#Execute: python infer_function_encode.py 2>&1 | tee submission1.txt

input_csv = '/home/donald/Desktop/test_out.csv'
missing_images_file = '/home/donald/Desktop/missing_indices.txt'

# just used for reading in an example image
image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
masks_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'

# set the path and name of the frozen model to load
froze_mod = '/home/donald/Desktop/PYTHON/kaggle_car_competition/model_12l_deconv/frozen.model'
# froze_mod = '/home/nes/Desktop/Caravana/frozen_models/5k-iter/frozen.model'
dsize = 6
# dosmooth = True

def load_frozen_model_remap_queue(frozen_model, input_image):
    # loads a frozen tensorflow model, remaps the input queue to input_image (queues overkill when inferring)
    model_file = open(frozen_model, 'rb')
    load_graph = tf.GraphDef()
    load_graph.ParseFromString(model_file.read())
    model_file.close()
    tf.import_graph_def(load_graph, input_map={"cv_MASTER_QUEUE/input_ims:0": input_image}, name='infer')
    print('frozen model loaded successfully from '+frozen_model)


def initialize_frozen_session(frozen_model):
    # this initializes a tensorflow session, restores the frozen model using the above function, then grabs the
    # tensorflow parameters that need to be fed in when running inference, also sets input_image used above
    with tf.Graph().as_default() as graph:
        input_images = tf.placeholder(tf.float32, shape=[None, 1280, 1918, 3], name='input_i')
        load_frozen_model_remap_queue(frozen_model, input_images)
    # snag the ops needed to actually run inference
    outputs = graph.get_tensor_by_name('infer/cv_CV_LAYERS/outputs:0')
    queue_select = graph.get_tensor_by_name('infer/cv_MASTER_QUEUE/select_queue:0')
    bsize = graph.get_tensor_by_name('infer/batch_size:0')
    dropout = graph.get_tensor_by_name('infer/dropout:0')
    # launch a session and run inference
    session = tf.Session(graph=graph)
    return input_images, outputs, queue_select, bsize, dropout, session


def infer_example(input_image, open_session, nnoutputs, input_images, queue_select, bsize, dropout):
    # runs inference on a single input image assuming initialize_frozen_session has already opened a tf session
    input_image = np.reshape(input_image, (1, 1280, 1918, 3))
    nnoutputs = open_session.run(nnoutputs, feed_dict={input_images: input_image, queue_select: 0, bsize: 1, dropout: 1.0})
    nnoutputs = np.reshape(nnoutputs, (1280, 1918))
    return nnoutputs


#-----------------------------------------------------------------------
#Encoding

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    im_list, all_masks, n_ims = inpipe.get_images_masks(image_dir, masks_dir)
    input_images, outputs, queue_select, bsize, dropout, session = initialize_frozen_session(froze_mod)
    missing_images = []
    missing_img_indices = []
    all_images = []
    readfile = open(input_csv, mode='r')
    counter = 0
    for line in readfile:
        if counter > 0:
            line = line.split(',')[0]
            all_images.append(line)
        counter += 1
    readfile.close()
    for i in range(len(im_list)):
        if im_list[i] not in all_images:
            missing_images.append(im_list[i])
            missing_img_indices.append(i)

    outputfile = open(input_csv, mode='a')
    indicesfile = open(missing_images_file, mode='w')
    for j in range(len(missing_img_indices)):
        indicesfile.write(str(missing_img_indices[j])+'\n')
    indicesfile.close()
    writer = csv.writer(outputfile, delimiter=',')
    for i in missing_img_indices:
        img, im_name = inpipe.not_random_image_reader(im_list, n_ims, 1.0, i)
        inferred = infer_example(img, session, outputs, input_images, queue_select, bsize, dropout)
        inferred_10 = np.around(inferred)
        selem = disk(dsize)
        inferred_10 = opening(inferred_10, selem)
        rle = rle_encode(inferred_10)
        rle_string = rle_to_string(rle)
        outputfile.write(im_name + ',' + rle_string + '\n')
    outputfile.close()





