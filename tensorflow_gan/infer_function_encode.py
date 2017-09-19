from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

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
        input_images = tf.placeholder(tf.float32, shape=[None, 1918, 1280, 3], name='input_i')
        load_frozen_model_remap_queue(frozen_model, input_images)
    # snag the ops needed to actually run inference
    outputs = graph.get_tensor_by_name('infer/cv_CV_LAYERS/outputs:0')
    queue_select = graph.get_tensor_by_name('infer/cv_MASTER_QUEUE/select_queue:0')
    bsize = graph.get_tensor_by_name('infer/batch_size:0')
    dropout = graph.get_tensor_by_name('infer/dropout:0')
    # launch a session and run inference
    session = tf.Session(graph=graph)
    return input_images, outputs, queue_select, bsize, dropout, session


def infer_example(input_image, open_session, outputs, input_images, queue_select, bsize, dropout):
    # runs inference on a single input image assuming initialize_frozen_session has already opened a tf session
    input_image = np.reshape(input_image, (1, 1918, 1280, 3))
    outputs = open_session.run(outputs, feed_dict={input_images: input_image, queue_select: 0, bsize: 1, dropout: 1.0})
    outputs = np.reshape(outputs, (1918, 1280))
    return outputs


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





if __name__ == '__main__':
    # EXAMPLE OF HOW TO RUN INFERENCE
    # these are just for benching/reading in an image
    import time
    import input_pipeline as inpipe

    # just used for reading in an example image
    image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
    masks_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'

    saved_csv = '/home/donald/Desktop/PYTHON/kaggle_car_competition/test_output.csv'

    # image_dir = '/home/nes/Desktop/Caravana/input/train/'
    # masks_dir = '/home/nes/Desktop/Caravana/input/train_masks/'

    # set the path and name of the frozen model to load
    froze_mod = '/home/donald/Desktop/temp/frozen.model'
    # froze_mod = '/home/nes/Desktop/Caravana/frozen_models/5k-iter/frozen.model'

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
    # for i in range(len(im_list)):
    # this for statement will just run on 0-999
    for i in range(len(im_list[0:1000])):
        img, mask, im_name = inpipe.not_random_image_reader(im_list, n_ims, 1.0, counter)
        # now feed the read-in image to infer_example along with initialized vars/session. Note that the read-in image
        # is assumed to be 1918x1280x3 numpy array with values normalized to range [0.0, 1.0]
        inferred = infer_example(img, session, outputs, input_images, queue_select, bsize, dropout)
        # note that inferred returns a 1918x1280 mask (float values, so will need to apply rounding to 0, 1)
        #print(np.shape(inferred))
        # now could call a function that adds the mask and image name to the output file to submit to kaggle
        # some_function()
        #------------------ encoding ----------------
        inferred_10 = np.around(inferred)
        rle = rle_encode(inferred_10)
        rle_string = rle_to_string(rle)
        #----------------------------------

        # write result to file
        output_file.write(im_name+','+rle_string+'\n')
        counter += 1

    # after the loop finishes you must close the tensorflow session to properly free hardware resources
    output_file.close()
    session.close()

    print('inference time: '+str(round(time.time()-init_split_time, 2)))


