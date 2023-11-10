import numpy as np
import re, os, traceback, sys, json
import tensorflow as tf
import timeit
from tensorflow.python.framework import ops

ops.reset_default_graph()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import flux
import lls_cnnmodel.training_model.ab_model

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tensor_regex = re.compile('.*:\d*')


# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name + ":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

def predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE=''):
    timer = timeit.default_timer()
    BATCH_SIZE = 4000
    # n_samples = np.array(flux).shape[1]
    n_samples = flux.shape[0]
    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred)  # set the 4 np matrix for every label

    with tf.Graph().as_default():
        lls_cnnmodel.training_model.ab_model.build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.compat.v1.Session() as sess:
            tf.compat.v1.train.Saver().restore(sess, checkpoint_filename + ".ckpt")
            for i in range(0, n_samples, BATCH_SIZE):
                pred[i:i + BATCH_SIZE], conf[i:i + BATCH_SIZE], offset[i:i + BATCH_SIZE], coldensity[i:i + BATCH_SIZE] = \
                    sess.run([t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                             feed_dict={t('x'): flux[i:i + BATCH_SIZE, :],
                                        t('keep_prob'): 1.0})

    print("Localize Model processed {:d} samples in chunks of {:d} in {:0.1f} seconds".format(
        n_samples, BATCH_SIZE, timeit.default_timer() - timer))

    # coldensity_rescaled = coldensity * COL_DENSITY_STD + COL_DENSITY_MEAN
    return pred, conf, offset, coldensity  # get the 4 labels prediction for every window


# Called from train_ann to perform a test of the train or test data, needs to separate pos/neg to get accurate #'s


if __name__ == '__main__':

    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty",
                       "dropout_keep_prob",
                       "fc1_n_neurons", "fc2_1_n_neurons", "fc2_2_n_neurons", "fc2_3_n_neurons",
                       "conv1_kernel", "conv2_kernel", "conv3_kernel", "conv4_kernel", "conv5_kernel", "conv6_kernel",
                       "conv1_filters", "conv2_filters", "conv3_filters", "conv4_filters", "conv5_filters",
                       "conv6_filters",
                       "conv1_stride", "conv2_stride", "conv3_stride", "conv4_stride", "conv5_stride", "conv6_stride",
                       "pool_kernel",
                       "pool_stride"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try
        # learning_rate
        [0.0006],
        # training_iters
        [50000],
        # batch_size
        [300],
        # l2_regularization_penalty
        [0.3],
        # dropout_keep_prob
        [0.9],
        # fc1_n_neurons
        [700],
        # fc2_1_n_neurons
        [200],
        # fc2_2_n_neurons
        [900],
        # fc2_3_n_neurons
        [700],

        # conv1_kernel
        [24],
        # conv2_kernel
        [16],
        # conv3_kernel
        [34],
        # conv4_kernel
        [3],
        # conv5_kernel
        [3],
        # conv6_kernel
        [3],

        # conv1_filters
        [100],
        # conv2_filters
        [64],
        # conv3_filters
        [64],
        # conv4_filters
        [256],
        # conv5_filters
        [128],
        # conv6_filters
        [80],

        # conv1_stride
        [4],
        # conv2_stride
        [1],
        # conv3_stride
        [1],
        # conv4_stride
        [3],
        # conv5_stride
        [2],
        # conv6_stride
        [1],

        # pool_kernel
        [4],
        # pool_stride
        [1]
    ]
    checkpoint_filename = 'model/current_99999'

    hyperparameters = {}
    for k in range(0, len(parameter_names)):
        hyperparameters[parameter_names[k]] = parameters[k][0]

    pred_dataset = 'sightlines.npy'  # the prediction data file
    r = np.load(pred_dataset, allow_pickle=True, encoding='latin1')  # .item()

    dataset = {}

    for sight_id in r.ravel():
        flux, lam = flux.make_dataset(sight_id)

        (pred, conf, offset, coldensity) = predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE='')

        dataset[sight_id.id] = {'pred': pred, 'conf': conf, 'offset': offset,
                             'coldensity': coldensity}  # save the window prediction as npy file, use it to get sightline prediction later

    np.save('pred.npy', dataset)

