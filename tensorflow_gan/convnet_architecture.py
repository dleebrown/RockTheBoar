import tensorflow as tf
import math

"""This will just be the neural network architecture

weights are initialized according to He et al. (2015)
biases are initialized to 0

architecture is 
training queue + cross validation queue assuming 3-color RGB images
3x3x3->16 conv layer (1918x1280x16)
ReLU
2x2x1 maxpooling (959x640x16)
3x3x16->128 (959x640x128)
ReLU
2x2x1 maxpooling (479x320x128)
3x3x128->256 (479x320x256)
ReLU
2x2x1 maxpooling (239x160x256)
flatten (9789440)
fully-connected layer 9789440 -> 4194304 (2^22)
dropout
ReLU
fully-connected layer 4194304 (2^20) -> 2455040

cross-entropy (softmax) cost
adam optimizer

need to work on how we want to do the output
"""

image_x = 1918
image_y = 1280
image_z = 3

# layer parameters
conv1_shape = [3, 3, 3, 16]
conv1_stride = [1, 1, 1, 1]
conv1_pstride = [1, 1, 1, 1]

conv2_shape = [3, 3, 16, 128]
conv2_stride = [1, 1, 1, 1]
conv2_pstride = [1, 1, 1, 1]

conv3_shape = [3, 3, 128, 256]
conv3_stride = [1, 1, 1, 1]
conv3_pstride = [1, 1, 1, 1]

fc1_shape = [9789440, 4194304]

fc2_shape = [fc1_shape[1], image_x*image_y]

# learning hyperparameters
learn_rate = 5.e-3
queue_depth = 100
batch_size = 10

########################################################################################################################
# functions


def initialize_conv_weights_bias(shape):
    """define the weight and bias initialization for conv layer. shape should be [x1, x2, x3, x4]
    """
    stdev = math.sqrt(2.0 / float(shape[1]*shape[2]))
    init_weight = tf.Variable(tf.random_normal(shape, mean=0.00, stddev=stdev))
    init_bias = tf.Variable(tf.constant(0.00, shape=[shape[3]]))
    return init_weight, init_bias


def initialize_fc_weights_bias(shape):
    """define the weight and bias initialization for fully connected layer. shape should be [x1, x2]
    """
    stdev = math.sqrt(2.0 / float(shape[0]))
    init_weight = tf.Variable(tf.random_normal(shape, mean=0.00, stddev=stdev))
    init_bias = tf.Variable(tf.constant(0.00, shape=shape[1]))
    return init_weight, init_bias


def conv_layer_block(inputs, weights, conv_stride, bias, pool_stride, maxpool=True):
    """The conv layer pattern is standardized so wrap it in a function
    """
    conv = tf.nn.conv2d(input=inputs, filter=weights, strides=conv_stride, padding='SAME')
    conv += bias
    activate = tf.nn.relu(conv)
    if maxpool:
        outputs = tf.nn.max_pool(value=activate, ksize=[1, 2, 2, 1], strides=pool_stride, padding='SAME')
    else:
        outputs = activate
    return outputs


def layer_flatten(inputs):
    shape = inputs.get_shape()
    nparams = shape[1:4].num_elements()
    outputs = tf.reshape(input,[-1,nparams])
    return outputs, nparams


def fc_layer_block(inputs, weights, bias, keep_prob, dropout=True, activate=True):
    """the fc layer pattern is also standardized"""
    connect = tf.matmul(inputs, weights)
    biased = connect+bias
    if dropout:
        dropped = tf.nn.dropout(biased, keep_prob)
    else:
        dropped = biased
    if activate:
        outputs = tf.nn.relu(dropped)
    else:
        outputs = dropped
    return outputs

########################################################################################################################
# input queue


prefix = 'cv_'
# placeholder to turn off dropout during inference
fc1_keep_prob = tf.placeholder(tf.float32, name='dropout1')
fc2_keep_prob = tf.placeholder(tf.float32, name='dropout2')


# input and output placeholders. note that known_bitmasks is already flattened
known_bitmasks = tf.placeholder(tf.float32, shape=[None, image_x*image_y], name='known_bitmasks')
image_data = tf.placeholder(tf.float32, shape=[None, image_x, image_y, image_z], name='image_data')
xval_bitmasks = tf.placeholder(tf.float32, shape=[None, image_x*image_y], name='xval_bitmasks')
xval_data = tf.placeholder(tf.float32, shape=[None, image_x, image_y, image_z], name='xval_data')

# input and cross-validation queue
with tf.name_scope('INPUT_QUEUE'):
    input_queue = tf.FIFOQueue(capacity=queue_depth, dtypes=[tf.float32, tf.float32],
                               shapes=[[image_x*image_y], [image_x, image_y, image_z]])
    queue_op = input_queue.enqueue_many([known_bitmasks, image_data])

# adding a second queue to handle xval data
with tf.name_scope('XVAL_QUEUE'):
    xval_queue = tf.FIFOQueue(capacity=queue_depth, dtypes=[tf.float32, tf.float32],
                              shapes=[[image_x*image_y], [image_x, image_y, image_z]])
    xval_op = xval_queue.enqueue_many([xval_bitmasks, xval_data])

# master queue that handles switching between the two input queues
with tf.name_scope('MASTER_QUEUE'):
    select_queue = tf.placeholder(tf.int32, [])
    master_queue = tf.QueueBase.from_list(select_queue, [input_queue, xval_queue])
    batch_of_outputs, batch_of_images = master_queue.dequeue_many(batch_size)

########################################################################################################################
# network architecture


# weights and bias initialization
with tf.name_scope(prefix+'WEIGHTS'):
    cweight1, cbias1 = initialize_conv_weights_bias(conv1_shape)
    cweight2, cbias2 = initialize_conv_weights_bias(conv2_shape)
    cweight3, cbias3 = initialize_conv_weights_bias(conv3_shape)
    fweight1, fbias1 = initialize_fc_weights_bias(fc1_shape)
    fweight2, fbias2 = initialize_fc_weights_bias(fc2_shape)


# layers
with tf.name_scope(prefix+'CV_LAYERS'):
    cv_block1 = conv_layer_block(batch_of_images, cweight1, conv1_stride, cbias1, conv1_pstride)
    cv_block2 = conv_layer_block(cv_block1, cweight2, conv2_stride, cbias2, conv2_pstride)
    cv_block3 = conv_layer_block(cv_block2, cweight3, conv3_stride, cbias3, conv3_pstride)

with tf.name_scope(prefix+'FLATTEN'):
    flattened, _ = layer_flatten(cv_block3)

with tf.name_scope(prefix='FC_LAYERS'):
    fc_block1 = fc_layer_block(flattened, fweight1, fbias1, fc1_keep_prob)
    network_outputs = fc_layer_block(fc_block1, fweight2, fbias2, fc2_keep_prob, dropout=False, activate=False)

# cost
batch_cost = tf.nn.softmax_cross_entropy_with_logits(logits=network_outputs, labels=batch_of_outputs)
mean_batch_cost = tf.reduce_mean(batch_cost)

# optimizer
optim_function = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(mean_batch_cost)

