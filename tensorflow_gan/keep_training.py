import numpy as np
import tensorflow as tf
import time
import input_pipeline as inpipe
import threading
import convnet_architecture_8l as cvarch
import matplotlib.pyplot as plt

# directory with images and masks
image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
masks_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'
# path to save the tensorflow and frozen models
save_model_path = '/home/donald/Desktop/PYTHON/kaggle_car_competition/model_8l_deconv/'

# number of training iterations
training_iterations = 100
# number of iterations before printing diagnostics like cost
sample_interval = 25
# control early stopping. this number is the max number of sample_intervals to go by with no improvement in cost
early_stop_threshold = 1500
# leave as false these are broken for now
use_xval = False
tboard_logging = False
# controls dropout - probability to retain a neuron
retain_prob = 0.80
# image scale factor - if using other than 1918x1280 adjust this. e.g. 959x640 -> scale factor = 0.5
scale_factor = 1.0
# number of threads to fetch training examples with - set to number of threads on cpu
num_threads = 4
# batch size - i can run with 2 on my 1060 for full size images. ultimately 1 is fine as long as you go enough iterations and turn down the
# learning rate to prevent big fluctuations between examples (true stochastic grad desc)
bsize = 1

im_list, all_masks, n_ims = inpipe.get_images_masks(image_dir, masks_dir)

# freezes a tensorflow model to be reloaded later
def freeze_model(save_dir):
    # the op to save - this results in all the tboard options being discarded from the frozen graph
    output_op = 'cv_COST/cost'
    frozen_dir = save_dir+'frozen.model'
    saver = tf.train.Saver()
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    session = tf.Session()
    # restore the trained model
    saver.restore(session, save_dir+'save.ckpt')
    # convert all useful variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(session, input_graph_def, [output_op])
    # open the specified file and write the model
    with open(frozen_dir, 'wb') as frozen_model:
        frozen_model.write(output_graph_def.SerializeToString())
    print('Model frozen as '+frozen_dir)

def add_to_queue(session, queue_operation, coordinator, list_of_images, list_of_masks, total_num_images, queuetype):
    img, msk = inpipe.random_image_reader(list_of_images, total_num_images, scale_factor)
    img = np.reshape(img, (1, 1280, 1918, 3), order='F')
    msk = np.reshape(msk, (1, 1280, 1918, 1), order='F')
    while not coordinator.should_stop():
            np.random.seed()
            if queuetype == 'train_q':
                try:
                    session.run(queue_operation, feed_dict={cvarch.image_data: img, cvarch.known_bitmasks: msk})
                except tf.errors.CancelledError:
                        print('Input queue closed, exiting training')
            if queuetype == 'xval_q':
                try:
                    session.run(queue_operation, feed_dict={cvarch.xval_data: img, cvarch.xval_params: msk})
                except tf.errors.CancelledError:
                        print('Input queue closed, exiting training')

def keep_training(total_iterations, keep_prob):
    # definition of the training loop
    def train_loop(iterations, learn_rate, keep_prob):
        begin_time = time.time()
        coordinator = tf.train.Coordinator()
        # force preprocessing to run on the cpu
        with tf.device("/cpu:0"):
            num_threads = 4
            queuetype = 'train_q'
            enqueue_threads = [threading.Thread(target=add_to_queue, args=(session, cvarch.queue_op, coordinator,
                                                                           im_list, all_masks, n_ims, queuetype))
                               for i in range(num_threads)]
            for i in enqueue_threads:
                i.start()
        feed_dict_train = {cvarch.dropout: keep_prob, cvarch.select_queue: 0, cvarch.batch_size: bsize}
        # controls early stopping threshold
        early_stop_counter = 0
        # stores best cost in order to control early stopping
        best_cost = 0.0
        # fetches early stop threshold if early stopping is enabled, otherwise just sets early stop iters to total iters
        # main training loop
        for i in range(iterations):
            # only continues if early stop threshold has not been met
            if early_stop_counter <= early_stop_threshold:
                # first iteration will store the cost under best_cost for xval if xval specified, current batch if not
                if i == 0:
                    #first_image = session.run(cvarch.image_data, feed_dict=feed_dict_train)
                    #print(np.shape(first_image))
                    init_cost = session.run(cvarch.mean_batch_cost, feed_dict=feed_dict_train)
                    best_cost = init_cost
                    print('Initial batch cost: '+str(round(init_cost, 2)))
                    session.run(cvarch.optim_function, feed_dict=feed_dict_train)
                # if not first iteration but iteration corresponding to sample interval, runs diagnostics
                elif (i+1) % int(sample_interval) == 0:
                    test_cost = session.run(cvarch.mean_batch_cost, feed_dict=feed_dict_train)
                    dice = session.run(cvarch.dice_val, feed_dict=feed_dict_train)
                    session.run(cvarch.optim_function, feed_dict=feed_dict_train)
                    # if xval set not specified, will just calculate cost for current batch and print
                    xvcost = test_cost
                    print('done with batch ' + str(int(i + 1)) + '/' + str(iterations) + ', current cost: '
                      + str(round(test_cost, 2)) + ', dice: ' + str(round(dice, 4)))
                    # if early stopping desired, will compare xval or current batch cost to the best cost
                    if float(xvcost) >= best_cost:
                        early_stop_counter += 1
                    else:
                        # reset early stopping counter if current cost is better than previous best cost
                        best_cost = xvcost
                        early_stop_counter = 0
                    if tboard_logging: # BROKEN
                        """
                        # if tensorboard logging enabled, stores visualization data
                        outlog = session.run(merged_summaries, feed_dict=feed_dict_train)
                        writer.add_summary(outlog, i+1)
                        """
                else:  # just runs optimize if none of the above criteria are met
                    session.run(cvarch.optim_function, feed_dict=feed_dict_train)
            if early_stop_counter == early_stop_threshold or (i == (iterations-1)
                                                              and early_stop_counter < (early_stop_threshold+1)):
                print('Early stop threshold or specified iterations met')
                early_stop_counter += 1
        # close the preprocessing queues and join threads
        coordinator.request_stop()
        session.run(cvarch.input_queue.close(cancel_pending_enqueues=True))
        if use_xval:
            session.run(cvarch.xval_queue.close(cancel_pending_enqueues=True))
        coordinator.join(enqueue_threads)
        end_time = time.time()
        # return the run time and total completed iterations
        return str(round(end_time-begin_time, 2))

    # control flow for training - load in save location, etc. launch tensorflow session, prepare saver
    model_dir = save_model_path
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, save_model_path + 'save.ckpt')
    print('model restored, continuing training...')
    # train the network on input training data
    execute_time = train_loop(total_iterations, cvarch.learn_rate, keep_prob)
    # save model and the graph and close session OFF FOR NOW
    save_path = saver.save(session, model_dir+'save.ckpt')
    session.close()
    print('Training finished in '+execute_time+'s')
                                               # , model and graph saved in '+save_path)
    # freeze the model and save it to disk after training # BROKEN
    freeze_model(save_model_path)

keep_training(training_iterations, retain_prob)