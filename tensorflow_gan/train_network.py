import numpy as np
import tensorflow as tf
import time
import input_pipeline as inpipe
import threading

image_dir = 'somedir'
masks_dir = 'somedir'

early_stop_threshold = 10000
sample_interval = 100
use_xval = False
tboard_logging = False

im_list, all_masks, n_ims = inpipe.get_images_masks(image_dir, masks_dir)

def add_to_queue(session, queue_operation, coordinator, list_of_images, list_of_masks, total_num_images, queuetype):
    while not coordinator.should_stop():
            np.random.seed()
            if queuetype == 'train_q':
                try:
                    session.run(queue_operation, feed_dict={image_data: proc_fluxes, known_params: known})
                except tf.errors.CancelledError:
                        print('Input queue closed, exiting training')
            if queuetype == 'xval_q':
                try:
                    session.run(queue_operation, feed_dict={xval_data: proc_fluxes, xval_params: known})
                except tf.errors.CancelledError:
                        print('Input queue closed, exiting training')

def train_neural_network(batch_size, iterations):
    # subloop definition to build a separate queue for xval data if it exists and calculate the cost
    def xval_subloop(learn_rate, bsize, step):
        queuetype = 'xval_q'
        xvcoordinator = tf.train.Coordinator()
        randomize = False
        # if xval preprocessing desired, flip the correct flat in add_to_queue
        if parameters['PREPROCESS_XVAL'] == 'YES':
            xval_training = True
        else:
            xval_training = False
        num_xvthread = int(parameters['XV_THREADS'])
        # force preprocessing to run on the cpu
        with tf.device("/cpu:0"):
            xval_threads = [threading.Thread(target=add_to_queue, args=(session, xval_op, xvcoordinator, xv_norm_out,
                                                                        xv_px_val, sn_range, interp_sn, y_off_range,
                                                                        xv_size, randomize, xval_training,
                                                                        queuetype)) for i in range(num_xvthread)]
            for i in xval_threads:
                i.start()
        feed_dict_xval = {learning_rate: learn_rate, dropout: 1.0, batch_size: bsize, select_queue: 1}
        xval_cost, xval_sum = session.run([cost, xval_cost_sum], feed_dict=feed_dict_xval)
        if parameters['TBOARD_OUTPUT'] == 'YES':
            writer.add_summary(xval_sum, step + inherit_iter_count)
        # close the queue and join threads
        xvcoordinator.request_stop()
        xvcoordinator.join(xval_threads)
        # return cost
        return round(xval_cost, 2)

    # definition of the training loop
    def train_loop(iterations, learn_rate, keep_prob, bsize):
        begin_time = time.time()
        coordinator = tf.train.Coordinator()
        # force preprocessing to run on the cpu
        with tf.device("/cpu:0"):
            num_threads = 12
            queuetype = 'train_q'
            enqueue_threads = [threading.Thread(target=add_to_queue, args=(session, queue_op, coordinator, im_list, all_masks, n_ims,
                                                                           queuetype)) for i in range(num_threads)]
            for i in enqueue_threads:
                i.start()
        feed_dict_train = {learning_rate: learn_rate, dropout: keep_prob, batch_size: bsize, select_queue: 0}
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
                        init_cost = xval_subloop(learn_rate, xv_size, i)
                        best_cost = init_cost
                        print('Initial xval cost: '+str(round(init_cost, 2)))
                        session.run(optimize, feed_dict=feed_dict_train)
                    else:
                        init_cost = session.run(cost, feed_dict=feed_dict_train)
                        best_cost = init_cost
                        print('Initial batch cost: '+str(round(init_cost, 2)))
                        session.run(optimize, feed_dict=feed_dict_train)
                # if not first iteration but iteration corresponding to sample interval, runs diagnostics
                elif (i+1) % int(sample_interval) == 0:
                    test_cost = session.run(cost, feed_dict=feed_dict_train)
                    session.run(optimize, feed_dict=feed_dict_train)
                    if use_xval:
                        xvcost = xval_subloop(learn_rate, xv_size, i)
                        print('done with batch '+str(int(i+1))+'/'+str(iterations)+', current cost: '
                          + str(round(test_cost, 2))+', xval cost: '+str(xvcost))
                    else:
                        # if xval set not specified, will just calculate cost for current batch and print
                        print('done with batch ' + str(int(i + 1)) + '/' + str(iterations) + ', current cost: '
                          + str(round(test_cost, 2)))
                    # if early stopping desired, will compare xval or current batch cost to the best cost
                    if float(xvcost) >= best_cost:
                        early_stop_counter += 1
                    else:
                        # reset early stopping counter if current cost is better than previous best cost
                        best_cost = xvcost
                        early_stop_counter = 0
                    if tboard_logging:
                        # if tensorboard logging enabled, stores visualization data
                        outlog = session.run(merged_summaries, feed_dict=feed_dict_train)
                        writer.add_summary(outlog, i+1+inherit_iter_count)
                else:  # just runs optimize if none of the above criteria are met
                    session.run(optimize, feed_dict=feed_dict_train)
            if early_stop_counter == early_stop_threshold or (i == (iterations-1)
                                                              and early_stop_counter < (early_stop_threshold+1)):
                # if end of training reached, print a message and optionally save timeline
                if parameters['TIMELINE_OUTPUT'] == 'YES':
                    # if timeline desired, prints to json file for most recent iteration
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(parameters['LOG_LOC'] + 'timeline_01.json', 'w') as f:
                        f.write(chrome_trace)
                    print('Timeline saved as timeline_01.json in folder ' + parameters['LOG_LOC'])
                print('Early stop threshold or specified iterations met')
                early_stop_counter += 1
        # close the preprocessing queues and join threads
        coordinator.request_stop()
        session.run(input_queue.close(cancel_pending_enqueues=True))
        if parameters['TRAINING_XVAL'] == 'YES':
            session.run(xval_queue.close(cancel_pending_enqueues=True))
        coordinator.join(enqueue_threads)
        end_time = time.time()
        # return the run time and total completed iterations
        return str(round(end_time-begin_time, 2)), completed_iterations

    # control flow for training - load in save location, etc. launch tensorflow session, prepare saver
    model_dir = parameters['SAVE_LOC']
    session = tf.Session()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # if tboard output desired start a writer
    if parameters['TBOARD_OUTPUT'] == 'YES':
        writer = tf.summary.FileWriter(parameters['LOG_LOC']+'logs', session.graph)
        merged_summaries = tf.summary.merge([cv1w_sum, cv1b_sum, cv1a_sum, fc1w_sum, fc1b_sum, fc1a_sum, fc2w_sum,
                                             fc2b_sum, fc2a_sum, fc3w_sum, fc3a_sum, batch_cost_sum])
    # train the network on input training data
    execute_time, finished_iters = train_loop(int(parameters['NUM_TRAIN_ITERS1']), float(parameters['LEARN_RATE1']),
                                              float(parameters['KEEP_PROB1']), bsize_train1, inherit_iter_count=0)
    # save model and the graph and close session
    save_path = saver.save(session, model_dir+'save.ckpt')
    session.close()
    print('Training stage 1 finished in '+execute_time+'s, model and graph saved in '+save_path)
    if parameters['DO_TRAIN2'] == 'YES':
        # if multistage training specified, repeat above process except for the metagraph saving
        bsize_train2 = int(parameters['BATCH_SIZE2'])
        print('Training stage 2 beginning, loading model...')
        session = tf.Session()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        saver = tf.train.Saver()
        saver.restore(session, model_dir+'save.ckpt')
        if parameters['TBOARD_OUTPUT'] == 'YES':
            writer = tf.summary.FileWriter(parameters['LOG_LOC'] + 'logs', session.graph)
            merged_summaries = tf.summary.merge([cv1w_sum, cv1b_sum, cv1a_sum, fc1w_sum, fc1b_sum, fc1a_sum, fc2w_sum,
                                                 fc2b_sum, fc2a_sum, fc3w_sum, fc3a_sum, batch_cost_sum])
        print('Model loaded, beginning training...')
        execute_time, _ = train_loop(int(parameters['NUM_TRAIN_ITERS2']), float(parameters['LEARN_RATE2']),
                                  float(parameters['KEEP_PROB2']), bsize_train2,
                                  inherit_iter_count=finished_iters)
        save_path = saver.save(session, model_dir)
        session.close()
        print('Training stage 2 finished in ' + execute_time + 's, model saved in ' + save_path)
    # freeze the model and save it to disk after training
    freeze_model(parameters)


