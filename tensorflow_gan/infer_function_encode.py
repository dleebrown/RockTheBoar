from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.morphology import opening
from skimage.morphology import disk
import multiprocessing
from functools import partial
import time
import input_pipeline as inpipe

# EXAMPLE FOR USE AT THE BOTTOM OF THIS FILE
#Execute: python infer_function_encode.py 2>&1 | tee submission1.txt

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

#-----------------------------------------------------------------------


# worker for multiprocessing
def mf_worker(imtuple, frozen_params):
    inferred = imtuple[0]
    name = imtuple[1]
    smooth = frozen_params[0]
    disksize = frozen_params[1]
    inferred_10 = np.around(inferred)
    if smooth:
        selem = disk(disksize)
        inferred_10 = opening(inferred_10, selem)
    rle = rle_encode(inferred_10)
    rle_string = rle_to_string(rle)
    return rle_string, name

if __name__ == '__main__':
    # EXAMPLE OF HOW TO RUN INFERENCE
    # these are just for benching/reading in an image


    # just used for reading in an example image
    image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
    masks_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'

    saved_csv = '/home/donald/Desktop/test_out.csv'

    # image_dir = '/home/nes/Desktop/Caravana/input/train/'
    # masks_dir = '/home/nes/Desktop/Caravana/input/train_masks/'

    # set the path and name of the frozen model to load
    froze_mod = '/home/donald/Desktop/PYTHON/kaggle_car_competition/model_8l_deconv/frozen.model'
    # froze_mod = '/home/nes/Desktop/Caravana/frozen_models/5k-iter/frozen.model'
    # number of images to run tensorflow on before starting the smoothing/rle encoding process
    qdepth = 100
    # number of processes to spawn for smoothing/rle encoding
    nprocesses = 6
    # control smoothing, and smoothing radius
    dsize = 6
    dosmooth = True

    # first thing to do is to run the initialize function to set up a session and retrieve graph variables
    # this is done outside of the inference loop so graph only loaded once
    input_images, outputs, queue_select, bsize, dropout, session = initialize_frozen_session(froze_mod)
    # benchmark - takes about 8 seconds to run this for 20 examples so maybe 6 hours for all test images
    init_split_time = time.time()
    # now within a loop the steps are read in an image (random training example used here)
    # then run infer_example using that image and the variables/session retrieved by initialization
    output_file = open(saved_csv, mode='w')
    output_file.write('img,rle_mask\n')
    # read in an image. for real inference these would be the test images read in one at a time within a loop
    im_list, all_masks, n_ims = inpipe.get_images_masks(image_dir, masks_dir)
    counter = 0
    # USE THIS FOR STATEMENT TO RUN ON ALL IMAGES
    #for i in range(len(im_list)):
    n_images_to_infer = len(im_list[0:10])
    for i in range(0, n_images_to_infer, qdepth):
        outputlist = []
        if (i+nprocesses <= n_images_to_infer) and (i+qdepth <= n_images_to_infer):
            for j in range(0, qdepth):
                img, im_name = inpipe.not_random_image_reader(im_list, n_ims, 1.0, counter)
                counter += 1
                # now feed the read-in image to infer_example along with initialized vars/session. Note that the read-in image
                # is assumed to be 1918x1280x3 numpy array with values normalized to range [0.0, 1.0]
                inferred = infer_example(img, session, outputs, input_images, queue_select, bsize, dropout)
                inputtuple = (inferred, im_name)
                outputlist.append(inputtuple)
        elif counter < n_images_to_infer:
            for j in range(0, n_images_to_infer-i):
                img, im_name = inpipe.not_random_image_reader(im_list, n_ims, 1.0, counter)
                counter += 1
                # now feed the read-in image to infer_example along with initialized vars/session. Note that the read-in image
                # is assumed to be 1918x1280x3 numpy array with values normalized to range [0.0, 1.0]
                inferred = infer_example(img, session, outputs, input_images, queue_select, bsize, dropout)
                inputtuple = (inferred, im_name)
                outputlist.append(inputtuple)
        frozen_params = (dosmooth, dsize)
        partial_worker = partial(mf_worker, frozen_params=frozen_params)
        # initializes the pool of processes
        pool = multiprocessing.Pool(nprocesses)
        # maps partial_worker and list of stars to the pool, stores used parameters in a list
        output_rle_name = pool.map(partial_worker, outputlist)
        # end the list of functions to go to pool
        pool.close()
        pool.join()
        # write result to file
        for k in range(0, len(output_rle_name)):
            output_file.write(output_rle_name[k][1]+','+output_rle_name[k][0]+'\n')
        counter += 1

    # after the loop finishes you must close the tensorflow session to properly free hardware resources
    output_file.close()
    session.close()
    print('inference time: '+str(round(time.time()-init_split_time, 2)))


