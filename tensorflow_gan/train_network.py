import numpy as np
import tensorflow as tf
import time
import input_pipeline as inpipe
import threading
import convnet_architecture as cvarch

image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
masks_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'
save_model_path = '/home/donald/Desktop/temp/'

training_iterations = 20000
early_stop_threshold = 500
sample_interval = 25
use_xval = False
tboard_logging = False
dropout_prob = 1.0
scale_factor = 0.5
num_threads = 4

im_list, all_masks, n_ims = inpipe.get_images_masks(image_dir, masks_dir)

def add_to_queue(session, queue_operation, coordinator, list_of_images, list_of_masks, total_num_images, queuetype):
    img, msk = inpipe.random_image_reader(list_of_images, total_num_images, scale_factor)
    img = np.reshape(img, (1, 959, 640, 3))
    msk = np.reshape(msk, (1, 959, 640, 1))
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

def train_network(total_iterations, keep_prob):
    def xval_subloop():
        queuetype = 'xval_q'
        xvcoordinator = tf.train.Coordinator()
        num_xvthread = 1
        # force preprocessing to run on the cpu
        with tf.device("/cpu:0"):
            xval_threads = [threading.Thread(target=add_to_queue, args=(session, cvarch.queue_op, xvcoordinator,
                                                                        im_list, all_masks, n_ims, queuetype))
                            for i in range(num_xvthread)]
            for i in xval_threads:
                i.start()
        feed_dict_xval = {cvarch.fc1_keep_prob: keep_prob, cvarch.fc2_keep_prob: keep_prob, cvarch.select_queue: 1}
        xval_cost, xval_dice = session.run([cvarch.mean_batch_cost, cvarch.dice_val], feed_dict=feed_dict_xval)
        """
        if parameters['TBOARD_OUTPUT'] == 'YES':
            writer.add_summary(xval_sum, step + inherit_iter_count)
        """
        # close the queue and join threads
        xvcoordinator.request_stop()
        xvcoordinator.join(xval_threads)
        # return cost, dice
        return round(xval_cost, 2), round(xval_dice, 4)

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
        feed_dict_train = {cvarch.fc1_keep_prob: keep_prob, cvarch.fc2_keep_prob: keep_prob, cvarch.select_queue: 0}
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
                    if use_xval:
                        init_cost, init_dice = xval_subloop()
                        best_cost = init_cost
                        print('Initial xval cost: '+str(round(init_cost, 2)))
                        session.run(cvarch.optim_function, feed_dict=feed_dict_train)
                    else:
                        init_cost = session.run(cvarch.mean_batch_cost, feed_dict=feed_dict_train)
                        best_cost = init_cost
                        print('Initial batch cost: '+str(round(init_cost, 2)))
                        session.run(cvarch.optim_function, feed_dict=feed_dict_train)
                # if not first iteration but iteration corresponding to sample interval, runs diagnostics
                elif (i+1) % int(sample_interval) == 0:
                    test_cost = session.run(cvarch.mean_batch_cost, feed_dict=feed_dict_train)
                    dice = session.run(cvarch.dice_val, feed_dict=feed_dict_train)
                    session.run(cvarch.optim_function, feed_dict=feed_dict_train)
                    if use_xval:
                        xvcost = xval_subloop()
                        print('done with batch '+str(int(i+1))+'/'+str(iterations)+', xval cost: '+str(xvcost))
                    else:
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
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # if tboard output desired start a writer
    if tboard_logging: # BROKEN
        """
        writer = tf.summary.FileWriter(parameters['LOG_LOC']+'logs', session.graph)
        merged_summaries = tf.summary.merge([cv1w_sum, cv1b_sum, cv1a_sum, fc1w_sum, fc1b_sum, fc1a_sum, fc2w_sum,
                                             fc2b_sum, fc2a_sum, fc3w_sum, fc3a_sum, batch_cost_sum])
        """
    # train the network on input training data
    execute_time = train_loop(total_iterations, cvarch.learn_rate, dropout_prob)
    # save model and the graph and close session OFF FOR NOW
    # save_path = saver.save(session, model_dir+'save.ckpt')
    session.close()
    print('Training finished in '+execute_time+'s')
                                               # , model and graph saved in '+save_path)
    # freeze the model and save it to disk after training # BROKEN
    # freeze_model(parameters)

train_network(training_iterations, keep_prob=1.0)


