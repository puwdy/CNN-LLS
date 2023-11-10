import tensorflow as tf


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape,
                                         stddev=0.1)  # Outputs random values from a truncated normal distribution，shape:The shape of the output tensor.stddev:The standard deviation of the normal distribution, before truncation
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)  # generating constant
    return tf.Variable(initial)


def conv1d(x, W, s):
    return tf.nn.conv2d(input=x, filters=W, strides=s,
                        padding='SAME')


def pooling_layer_parameterized(pool_method, h_conv, pool_kernel, pool_stride):
    if pool_method == 1:
        return tf.nn.max_pool2d(input=h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1],
                                padding='SAME')
    elif pool_method == 2:
        return tf.nn.avg_pool2d(input=h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1],
                                padding='SAME')


def variable_summaries(var, name, collection):
    # Attach a lot of summaries to a Tensor.
    with tf.compat.v1.name_scope('summaries') as r:
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.add_to_collection(collection, tf.compat.v1.summary.scalar('mean/' + name, mean))
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.add_to_collection(collection, tf.compat.v1.summary.scalar('stddev/' + name, stddev))
        tf.compat.v1.add_to_collection(collection,
                                       tf.compat.v1.summary.scalar('max/' + name, tf.reduce_max(input_tensor=var)))
        tf.compat.v1.add_to_collection(collection,
                                       tf.compat.v1.summary.scalar('min/' + name, tf.reduce_min(input_tensor=var)))
        tf.compat.v1.add_to_collection(collection, tf.compat.v1.summary.histogram(name, var))


def build_model(hyperparameters):
    import math

    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    l2_regularization_penalty = hyperparameters['l2_regularization_penalty']
    fc1_n_neurons = hyperparameters['fc1_n_neurons']
    fc2_1_n_neurons = hyperparameters['fc2_1_n_neurons']
    fc2_2_n_neurons = hyperparameters['fc2_2_n_neurons']
    fc2_3_n_neurons = hyperparameters['fc2_3_n_neurons']
    conv1_kernel = hyperparameters['conv1_kernel']
    conv2_kernel = hyperparameters['conv2_kernel']
    conv3_kernel = hyperparameters['conv3_kernel']
    conv4_kernel = hyperparameters['conv4_kernel']
    conv5_kernel = hyperparameters['conv5_kernel']
    conv6_kernel = hyperparameters['conv6_kernel']

    conv1_filters = hyperparameters['conv1_filters']
    conv2_filters = hyperparameters['conv2_filters']
    conv3_filters = hyperparameters['conv3_filters']
    conv4_filters = hyperparameters['conv4_filters']
    conv5_filters = hyperparameters['conv5_filters']
    conv6_filters = hyperparameters['conv6_filters']

    conv1_stride = hyperparameters['conv1_stride']
    conv2_stride = hyperparameters['conv2_stride']
    conv3_stride = hyperparameters['conv3_stride']
    conv4_stride = hyperparameters['conv4_stride']
    conv5_stride = hyperparameters['conv5_stride']
    conv6_stride = hyperparameters['conv6_stride']

    pool_kernel = hyperparameters['pool_kernel']
    pool_stride = hyperparameters['pool_stride']
    pool_method = 1

    INPUT_SIZE = 400
    tfo = {}  # Tensorflow objects
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
    label_classifier = tf.compat.v1.placeholder(tf.float32, shape=[None], name='label_classifier')  # 1D
    label_offset = tf.compat.v1.placeholder(tf.float32, shape=[None], name='label_offset')
    label_coldensity = tf.compat.v1.placeholder(tf.float32, shape=[None], name='label_coldensity')
    keep_prob = tf.compat.v1.placeholder(tf.float32,
                                         name='keep_prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    x_4d = tf.reshape(x, [-1, INPUT_SIZE, 1, 1])

    W_conv1 = weight_variable([conv1_kernel, 1, 1, conv1_filters])
    b_conv1 = bias_variable([conv1_filters])
    stride1 = [1, conv1_stride, 1, 1]
    h_conv1 = tf.nn.relu(conv1d(x_4d, W_conv1, stride1) + b_conv1)

    W_conv2 = weight_variable([conv2_kernel, 1, conv1_filters, conv2_filters])
    b_conv2 = bias_variable([conv2_filters])
    stride2 = [1, conv2_stride, 1, 1]
    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2, stride2) + b_conv2)

    W_conv3 = weight_variable([conv3_kernel, 1, conv2_filters, conv3_filters])
    b_conv3 = bias_variable([conv3_filters])
    stride3 = [1, conv3_stride, 1, 1]
    h_conv3 = tf.nn.relu(conv1d(h_conv2, W_conv3, stride3) + b_conv3)

    W_conv4 = weight_variable([conv4_kernel, 1, conv3_filters, conv4_filters])
    b_conv4 = bias_variable([conv4_filters])
    stride4 = [1, conv4_stride, 1, 1]
    h_conv4 = tf.nn.relu(conv1d(h_conv3, W_conv4, stride4) + b_conv4)

    # Third convolutional layer
    W_conv5 = weight_variable([conv5_kernel, 1, conv4_filters, conv5_filters])
    b_conv5 = bias_variable([conv5_filters])
    stride5 = [1, conv5_stride, 1, 1]
    h_conv5 = tf.nn.relu(conv1d(h_conv4, W_conv5, stride5) + b_conv5)

    W_conv6 = weight_variable([conv6_kernel, 1, conv5_filters, conv6_filters])
    b_conv6 = bias_variable([conv6_filters])
    stride6 = [1, conv6_stride, 1, 1]
    h_conv6 = tf.nn.relu(conv1d(h_conv5, W_conv6, stride6) + b_conv6)
    h_pool = pooling_layer_parameterized(pool_method, h_conv6, pool_kernel, pool_stride)

    # FC1: first fully connected layer, shared
    inputsize_fc1 = int(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(math.ceil(
        INPUT_SIZE / conv1_stride) / conv2_stride) / conv3_stride) / conv5_stride) / conv6_stride) / conv4_stride) / pool_stride)) * conv4_filters
    # batch_size = x.get_shape().as_list()[0]
    h_pool3_flat = tf.reshape(h_pool, [-1, inputsize_fc1])
    W_fc1 = weight_variable([inputsize_fc1, fc1_n_neurons])
    b_fc1 = bias_variable([fc1_n_neurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout FC1
    h_fc1_drop = tf.nn.dropout(h_fc1, 1 - (keep_prob))
    W_fc2_1 = weight_variable([fc1_n_neurons, fc2_1_n_neurons])
    b_fc2_1 = bias_variable([fc2_1_n_neurons])
    W_fc2_2 = weight_variable([fc1_n_neurons, fc2_2_n_neurons])
    b_fc2_2 = bias_variable([fc2_2_n_neurons])
    W_fc2_3 = weight_variable([fc1_n_neurons, fc2_3_n_neurons])
    b_fc2_3 = bias_variable([fc2_3_n_neurons])

    # FC2 activations
    h_fc2_1 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_1) + b_fc2_1)  #
    h_fc2_2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_2) + b_fc2_2)
    h_fc2_3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_3) + b_fc2_3)

    # FC2 Dropout [1-3]
    h_fc2_1_drop = tf.nn.dropout(h_fc2_1, 1 - (keep_prob))
    h_fc2_2_drop = tf.nn.dropout(h_fc2_2, 1 - (keep_prob))
    h_fc2_3_drop = tf.nn.dropout(h_fc2_3, 1 - (keep_prob))

    # Readout Layer
    W_fc3_1 = weight_variable([fc2_1_n_neurons, 1])
    b_fc3_1 = bias_variable([1])
    W_fc3_2 = weight_variable([fc2_2_n_neurons, 1])
    b_fc3_2 = bias_variable([1])
    W_fc3_3 = weight_variable([fc2_3_n_neurons, 1])
    b_fc3_3 = bias_variable([1])

    y_fc4_1 = tf.add(tf.matmul(h_fc2_1_drop, W_fc3_1), b_fc3_1)  # w*x+b
    y_nn_classifier = tf.reshape(y_fc4_1, [-1], name='y_nn_classifer')

    y_fc4_2 = tf.add(tf.matmul(h_fc2_2_drop, W_fc3_2), b_fc3_2)
    y_nn_offset = tf.reshape(y_fc4_2, [-1], name='y_nn_offset')
    y_fc4_3 = tf.add(tf.matmul(h_fc2_3_drop, W_fc3_3), b_fc3_3)
    y_nn_coldensity = tf.reshape(y_fc4_3, [-1], name='y_nn_coldensity')

    # Train and Evaluate the model
    loss_classifier = tf.add(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_nn_classifier, labels=label_classifier),
                             l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                          tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
                             name='loss_classifier')
    #
    loss_offset_regression = tf.add(tf.reduce_sum(input_tensor=tf.nn.l2_loss(y_nn_offset - label_offset)),
                                    l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                                 tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_2)),
                                    name='loss_offset_regression')
    epsilon = 1e-6
    loss_coldensity_regression = tf.reduce_sum(
        input_tensor=tf.multiply(tf.square(y_nn_coldensity - label_coldensity),
                                 tf.compat.v1.div(label_coldensity, label_coldensity + epsilon)) +
                     l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
        name='loss_coldensity_regression')

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    cost_pos_samples_lossfns_ABC = loss_classifier + loss_offset_regression + loss_coldensity_regression
    train_step_ABC = optimizer.minimize(cost_pos_samples_lossfns_ABC, global_step=global_step, name='train_step_ABC')
    output_classifier = tf.sigmoid(y_nn_classifier, name='output_classifier')
    prediction = tf.round(output_classifier, name='prediction')
    correct_prediction = tf.equal(prediction, label_classifier)
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32), name='accuracy')
    rmse_offset = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(tf.subtract(y_nn_offset, label_offset))),
                          name='rmse_offset')
    rmse_coldensity = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(tf.subtract(y_nn_coldensity, label_coldensity))),
                              name='rmse_coldensity')
    variable_summaries(loss_classifier, 'loss_classifier', 'SUMMARY_A')
    variable_summaries(loss_offset_regression, 'loss_offset_regression', 'SUMMARY_B')
    variable_summaries(loss_coldensity_regression, 'loss_coldensity_regression', 'SUMMARY_C')
    variable_summaries(accuracy, 'classification_accuracy', 'SUMMARY_A')
    variable_summaries(rmse_offset, 'rmse_offset', 'SUMMARY_B')
    variable_summaries(rmse_coldensity, 'rmse_coldensity', 'SUMMARY_C')

    return train_step_ABC, tfo
