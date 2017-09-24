import tensorflow as tf
import math
import numpy as np

# biases initialized to 0, weights initialized to He et al. (2015).
# note that dropout is on by default on all layers but the last
# these all need to be set
image_y = 1918
image_x = 1280
image_z = 3

# layer parameters
conv1_shape = [15, 15, 3, 8]
conv1_stride = [1, 1, 1, 1]
conv1_pstride = [1, 2, 2, 1]

conv2_shape = [7, 7, conv1_shape[3], 16]
conv2_stride = [1, 1, 1, 1]
conv2_pstride = [1, 2, 2, 1]

conv3_shape = [3, 3, conv2_shape[3], 16]
conv3_stride = [1, 1, 1, 1]
conv3_pstride = [1, 1, 1, 1]

conv4_shape = [3, 3, conv3_shape[3], 32]
conv4_stride = [1, 1, 1, 1]
conv4_pstride = [1, 1, 1, 1]

conv5_shape = [3, 3, conv4_shape[3], 64]
conv5_stride = [1, 1, 1, 1]
conv5_pstride = [1, 1, 1, 1]

conv6_shape = [3, 3, conv5_shape[3], 128]
conv6_stride = [1, 1, 1, 1]
conv6_pstride = [1, 1, 1, 1]

conv7_shape = [3, 3, conv6_shape[3], 256]
conv7_stride = [1, 1, 1, 1]
conv7_pstride = [1, 1, 1, 1]

conv8_shape = [3, 3, conv7_shape[3], 512]
conv8_stride = [1, 1, 1, 1]
conv8_pstride = [1, 1, 1, 1]

dc1_shape = [3, 3, 256, conv8_shape[3]]
dc1_stride = [1, 1, 1, 1]
dc1_output_shape = [1, 320, 480, dc1_shape[2]]

dc2_shape = [3, 3, 128, dc1_output_shape[3]]
dc2_stride = [1, 1, 1, 1]
dc2_output_shape = [1, 320, 480, dc2_shape[2]]

dc3_shape = [3, 3, 64, dc2_output_shape[3]]
dc3_stride = [1, 2, 2, 1]
dc3_output_shape = [1, 640, 959, dc3_shape[2]]

dc4_shape = [3, 3, 1, dc3_output_shape[3]]
dc4_stride = [1, 2, 2, 1]
dc4_output_shape = [1, 1280, 1918, dc4_shape[2]]

# learning hyperparameters
learn_rate = 1.e-4
queue_depth = 25

########################################################################################################################
# functions


def initialize_conv_weights_bias(shape):
    """define the weight and bias initialization for conv layer. shape should be [x1, x2, x3, x4]
    """
    stdev = math.sqrt(2.0 / float(shape[0]*shape[1]*shape[3]))
    init_weight = tf.Variable(tf.random_normal(shape, mean=0.00, stddev=stdev))
    init_bias = tf.Variable(tf.constant(0.00, shape=[shape[3]]))
    return init_weight, init_bias


def initialize_deconv_weights_bias(shape):
    """define the weight and bias initialization for conv layer. shape should be [x1, x2, x3, x4]
    """
    stdev = math.sqrt(2.0 / float(shape[0]*shape[1]*shape[3]))
    init_weight = tf.Variable(tf.random_normal(shape, mean=0.00, stddev=stdev))
    init_bias = tf.Variable(tf.constant(0.00, shape=[shape[2]]))
    return init_weight, init_bias


def initialize_fc_weights_bias(shape):
    """define the weight and bias initialization for fully connected layer. shape should be [x1, x2]
    """
    stdev = math.sqrt(2.0 / float(shape[0]))
    init_weight = tf.Variable(tf.random_normal(shape, mean=0.00, stddev=stdev))
    init_bias = tf.Variable(tf.constant(0.00, shape=[shape[1]]))
    return init_weight, init_bias


def conv_layer_block(inputs, weights, conv_stride, bias, pool_stride, keep_prob, maxpool=True, drop=True, activation='relu'):
    """The conv layer pattern is standardized so wrap it in a function
    """
    conv = tf.nn.conv2d(input=inputs, filter=weights, strides=conv_stride, padding='SAME')
    conv += bias
    if activation == 'relu':
        activate = tf.nn.relu(conv)
    else:
        activate = tf.nn.sigmoid(conv)
    if drop:
        dropped = tf.nn.dropout(activate, keep_prob)
    else:
        dropped = activate
    if maxpool:
        outputs = tf.nn.max_pool(value=dropped, ksize=[1, 2, 2, 1], strides=pool_stride, padding='SAME')
    else:
        outputs = dropped
    return outputs


def deconv_layer_block(inputs, weights, bias, output_shape, stride, activation='relu'):
    deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride, padding='SAME')
    deconv += bias
    if activation == 'relu':
        activate = tf.nn.relu(deconv)
    else:
        activate = tf.nn.sigmoid(deconv)
    return activate

smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = tf.contrib.layers.flatten(y_true)
    y_pred_f = tf.contrib.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y_true) - dice_coef(y_true, y_pred)

########################################################################################################################
# input queue

prefix = 'cv_'
# placeholder to turn off dropout during inference
dropout = tf.placeholder(tf.float32, name='dropout')
batch_size = tf.placeholder(tf.int32, name='batch_size')


# input and output placeholders. note that known_bitmasks is already flattened
known_bitmasks = tf.placeholder(tf.float32, shape=[None, image_x, image_y, 1], name='known_bitmasks')
image_data = tf.placeholder(tf.float32, shape=[None, image_x, image_y, image_z], name='image_data')
xval_bitmasks = tf.placeholder(tf.float32, shape=[None, image_x, image_y, 1], name='xval_bitmasks')
xval_data = tf.placeholder(tf.float32, shape=[None, image_x, image_y, image_z], name='xval_data')

# input and cross-validation queue
with tf.name_scope('INPUT_QUEUE'):
    input_queue = tf.FIFOQueue(capacity=queue_depth, dtypes=[tf.float32, tf.float32],
                               shapes=[[image_x, image_y, 1], [image_x, image_y, image_z]])
    queue_op = input_queue.enqueue_many([known_bitmasks, image_data])

# adding a second queue to handle xval data
with tf.name_scope('XVAL_QUEUE'):
    xval_queue = tf.FIFOQueue(capacity=queue_depth, dtypes=[tf.float32, tf.float32],
                              shapes=[[image_x, image_y, 1], [image_x, image_y, image_z]])
    xval_op = xval_queue.enqueue_many([xval_bitmasks, xval_data])

# master queue that handles switching between the two input queues
with tf.name_scope(prefix+'MASTER_QUEUE'):
    select_queue = tf.placeholder(tf.int32, [], name='select_queue')
    master_queue = tf.QueueBase.from_list(select_queue, [input_queue, xval_queue])
    batch_of_outputs, batch_of_ims = master_queue.dequeue_many(batch_size)
    # identity functions to support naming to be called during inference
    batch_of_images = tf.identity(batch_of_ims, name='input_ims')

########################################################################################################################
# network architecture


# weights and bias initialization
with tf.name_scope(prefix+'WEIGHTS'):
    cweight1, cbias1 = initialize_conv_weights_bias(conv1_shape)
    cweight2, cbias2 = initialize_conv_weights_bias(conv2_shape)
    cweight3, cbias3 = initialize_conv_weights_bias(conv3_shape)
    cweight4, cbias4 = initialize_conv_weights_bias(conv4_shape)
    cweight5, cbias5 = initialize_conv_weights_bias(conv5_shape)
    cweight6, cbias6 = initialize_conv_weights_bias(conv6_shape)
    cweight7, cbias7 = initialize_conv_weights_bias(conv7_shape)
    cweight8, cbias8 = initialize_conv_weights_bias(conv8_shape)
    dcweight1, dcbias1 = initialize_deconv_weights_bias(dc1_shape)
    dcweight2, dcbias2 = initialize_deconv_weights_bias(dc2_shape)
    dcweight3, dcbias3 = initialize_deconv_weights_bias(dc3_shape)
    dcweight4, dcbias4 = initialize_deconv_weights_bias(dc4_shape)


# layers
with tf.name_scope(prefix+'CV_LAYERS'):
    cv_block1 = conv_layer_block(batch_of_images, cweight1, conv1_stride, cbias1, conv1_pstride, dropout, drop=True,
                                 maxpool=True)
    cv_block2 = conv_layer_block(cv_block1, cweight2, conv2_stride, cbias2, conv2_pstride, dropout, drop=True,
                                 maxpool=True)
    cv_block3 = conv_layer_block(cv_block2, cweight3, conv3_stride, cbias3, conv3_pstride, dropout, drop=True,
                                 maxpool=False)
    cv_block4 = conv_layer_block(cv_block3, cweight4, conv4_stride, cbias4, conv4_pstride, dropout, drop=True,
                                 maxpool=False)
    cv_block5 = conv_layer_block(cv_block4, cweight5, conv5_stride, cbias5, conv5_pstride, dropout, drop=True,
                                 maxpool=False)
    cv_block6 = conv_layer_block(cv_block5, cweight6, conv6_stride, cbias6, conv6_pstride, dropout, drop=True,
                                 maxpool=False)
    cv_block7 = conv_layer_block(cv_block6, cweight7, conv7_stride, cbias7, conv7_pstride, dropout, drop=True,
                                 maxpool=False)
    cv_block8 = conv_layer_block(cv_block7, cweight8, conv8_stride, cbias8, conv8_pstride, dropout, drop=True,
                                 maxpool=False)
    dc_block1 = deconv_layer_block(cv_block8, dcweight1, dcbias1, dc1_output_shape, dc1_stride)
    dc_block2 = deconv_layer_block(dc_block1, dcweight2, dcbias2, dc2_output_shape, dc2_stride)
    dc_block3 = deconv_layer_block(dc_block2, dcweight3, dcbias3, dc3_output_shape, dc3_stride)
    dc_block4 = deconv_layer_block(dc_block3, dcweight4, dcbias4, dc4_output_shape, dc4_stride, activation='sigmoid')
    network_outputs = tf.identity(dc_block4, name='outputs')

# cost
"""
batch_cost = tf.nn.softmax_cross_entropy_with_logits(logits=network_outputs, labels=batch_of_outputs)
mean_batch_cost = tf.reduce_mean(batch_cost)
"""

dice_val = dice_coef(batch_of_outputs, network_outputs)

batch_cost = bce_dice_loss(batch_of_outputs, network_outputs)

with tf.name_scope(prefix+'COST'):
    mean_batch_cost = tf.reduce_mean(batch_cost, name='cost')

# optimizer
optim_function = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(mean_batch_cost)

