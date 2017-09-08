import tensorflow as tf
import numpy as np

def load_frozen_model_remap_queue(frozen_model, input_image):
    model_file = open(frozen_model, 'rb')
    load_graph = tf.GraphDef()
    load_graph.ParseFromString(model_file.read())
    model_file.close()
    tf.import_graph_def(load_graph, input_map={"cv_MASTER_QUEUE/input_ims:0": input_image}, name='infer')
    print('frozen model loaded successfully from '+frozen_model)


def inference(input_image, frozen_model):
    input_image = np.reshape(input_image, (1, 959, 640, 3))
    with tf.Graph().as_default() as graph:
        input_images = tf.placeholder(tf.float32, shape=[None, 959, 640, 3], name='input_i')
        load_frozen_model_remap_queue(frozen_model, input_images)
    # snag the ops needed to actually run inference
    outputs = graph.get_tensor_by_name('infer/cv_CV_LAYERS/outputs:0')
    queue_select = graph.get_tensor_by_name('infer/cv_MASTER_QUEUE/select_queue:0')
    bsize = graph.get_tensor_by_name('infer/batch_size:0')
    """
    dropout = graph.get_tensor_by_name('infer/NETWORK_HYPERPARAMS/dropout:0')
    batch_size = graph.get_tensor_by_name('infer/NETWORK_HYPERPARAMS/batch_size:0')
    """
    # launch a session and run inference
    session = tf.Session(graph=graph)
    outputs = session.run(outputs, feed_dict={input_images: input_image, queue_select: 0, bsize: 1})
    session.close()
    return outputs

if __name__ == '__main__':
    import input_pipeline as inpipe

    image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
    masks_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train_masks/'
    froze_mod = '/home/donald/Desktop/temp/frozen.model'

    im_list, all_masks, n_ims = inpipe.get_images_masks(image_dir, masks_dir)
    img, mask = inpipe.random_image_reader(im_list, n_ims, scale_factor=0.5)
    print(np.shape(img))
    inferred = inference(img, froze_mod)
    print(inferred)

