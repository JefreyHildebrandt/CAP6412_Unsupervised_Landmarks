from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
sess =  tf.InteractiveSession()

def get_estimator(distort, orig):
    
    features_network = tf.estimator.Estimator(model_fn=get_conv_features, model_dir="/tmp/stuff", params={'original_images':orig})#, 'global_step':tf.Variable(0, trainable=False)})

    # dataset_distort = tf.contrib.data.Dataset.from_tensor_slices(distort) 
    # dataset_orig = tf.contrib.data.Dataset.from_tensor_slices(orig)

    #DEBUG
    # with tf.Session() as sess:
    #   tf.global_variables_initializer()
    #   sess.run(get_conv_features({'dataset_distort': tf.convert_to_tensor(distort)}, tf.convert_to_tensor(orig), tf.estimator.ModeKeys.TRAIN, {'original_images':orig}, None))
    # get_conv_features({'dataset_distort': tf.convert_to_tensor(distort)}, tf.convert_to_tensor(orig), tf.estimator.ModeKeys.TRAIN, {'original_images':orig}, None)
    #END DEBUG

    train_input_fn =  tf.estimator.inputs.numpy_input_fn(
      x={"dataset_distort": distort},
      y=orig,
      batch_size=50,
      num_epochs=None,
      shuffle=True)
    
    features_network.train(train_input_fn, steps=1)
    # predictions = features_network.predict(train_input_fn)
    # y_predicted = np.array(list(p['gauss_x'] for p in predictions))
    # y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # y_predicted_two = np.array(list(p['gauss_y'] for p in predictions))
    # y_predicted_two = y_predicted_two.reshape(np.array(y_test).shape)
    print()

def get_conv_features(features, labels, mode, params, config):
    K = 10
    # input_shape = [10, 128, 128, 3]
    input_shape = features['dataset_distort']
    # global_step = params['global_step']
    # num_blocks = 4
    cur_feature_channels = 32
    start_conv = get_next_conv_block(input_shape, cur_feature_channels, kernel_size=[7, 7], strides=1)

    cur_feature_channels *= 2
    conv_two = get_next_conv_block(start_conv, cur_feature_channels)

    cur_feature_channels *= 2
    conv_three = get_next_conv_block(conv_two, cur_feature_channels)

    cur_feature_channels *= 2
    last_conv = get_next_conv_block(conv_three, cur_feature_channels)

    last_conv_shape = last_conv.shape.as_list()

    heatmaps = tf.layers.conv2d(inputs=last_conv, filters=K, kernel_size=[1, 1], strides=1)

    gauss_y, gauss_y_prob = get_coordinates_from_heatmaps(heatmaps, 2, last_conv_shape[1])  # B,NMAP
    gauss_x, gauss_x_prob = get_coordinates_from_heatmaps(heatmaps, 1, last_conv_shape[2])  # B,NMAP

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
        'gauss_x': gauss_x,
        'gauss_x_prob': gauss_x_prob,
        'gauss_y': gauss_y,
        'gauss_y_prob': gauss_y_prob,
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   # # sess.run(gauss_y)
    #   y = gauss_y.eval()
    #   # print()
    #   ngauss = tf.get_variable("gauss_y", shape=gauss_y.shape, initializer=tf.zeros_initializer())
    #   assignment = ngauss.assign_add(gauss_y)
    #   with tf.control_dependencies([assignment]):
    #     val = ngauss.read_value()
    #     print(val)
      # tf.global_variables_initializer().run()
      # val = sess.run(assignment)
      # print(val)
      # print()

    gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)

    
    # gauss_xy = []
    # for map_size in [16, 32, 64, 128]:
    gauss_map = get_gaussian_maps(gauss_mu, [last_conv_shape[1], last_conv_shape[2]], 10.0)
        # gauss_xy.append(gauss_map)
    # LOOK AT SIMPLE_RENDERER AND NOTICE THAT ONLY THE SMALLEST SIZE GAUSSIAN MAP IS NEEDED
    combined_renderer_input = tf.concat([last_conv, gauss_map], axis=-1)
    

    print()

    rebuild_image_conv_net(combined_renderer_input)
    # prev_layer = start_conv
    # for _ in range(1, num_blocks):
    #     cur_feature_channels *= 2
    #     prev_layer = get_next_conv_block(prev_layer, cur_feature_channels)
    # with 3 layers with stride of 2, with input 128x128x3 the output will be 16x16xN
    lr = tf.train.exponential_decay(0.001,
                              tf.train.get_global_step(),
                              100000,
                              0.95,
                              staircase=True)
    loss = []
    train_op = tf.train.AdamOptimizer(learning_rate=lr).compute_gradients(loss, colocate_gradients_with_ops=True)



    return tf.estimator.EstimatorSpec(mode=mode, predictions=None, loss=loss, train_op=train_op)

def get_coordinates_from_heatmaps(heatmaps, x_y, x_y_size):
    g_c_prob = tf.reduce_mean(heatmaps, axis=x_y)  # B,W,NMAP
    # g_c_prob = tf.exp(g_c_prob) / tf.reduce_sum(tf.exp(g_c_prob), x_y)
    g_c_prob = tf.nn.softmax(g_c_prob, dim=1)  # B,W,NMAP
    coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, x_y_size)) # W
    coord_pt = tf.reshape(coord_pt, [1, x_y_size, 1])
    g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
    return g_c, g_c_prob

def get_gaussian_maps(mu, shape_hw, inv_std):
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

    x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))

    mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

    y = tf.reshape(y, [1, 1, shape_hw[0], 1])
    x = tf.reshape(x, [1, 1, 1, shape_hw[1]])

    g_y = tf.square(y - mu_y)
    g_x = tf.square(x - mu_x)
    dist = (g_y + g_x) * inv_std**2
    g_yx = tf.exp(-dist)

    g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
    return g_yx


def get_next_conv_block(inputs_layer, filter_size, kernel_size=[3, 3], strides=2):
    conv1 = tf.layers.conv2d(inputs=inputs_layer, filters=filter_size, kernel_size=kernel_size, strides=strides)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=filter_size, kernel_size=[3, 3], strides=1)
    return conv2

def rebuild_image_conv_net(inputs_layer, output_shape=[128, 128, 3]):
    pass