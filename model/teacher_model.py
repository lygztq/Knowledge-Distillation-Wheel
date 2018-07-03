import tensorflow as tf
import numpy as np
from model.maxout import maxout

def cifar10_teacher_model(input_tensor, trainable=True, is_train=True, temp=1.0):
    """
    This teacher model is defined in http://arxiv.org/abs/1412.6550 and Maxout paper.
    The teacher model is composed of three conv layers with maxout activation function
    and max-pooling. The last conv layer is followed by a FC-maxout layer of 500 units and a 
    top softmax layer.

    layer1: (WxHx3) -> [ conv(3x192x8x8) --(WxHx192)-> maxout(2) --(WxHx96)-> max-pooling(4x4, 2) ] -> ([W/2]x[H/2]x96)
    layer2: ([W/4]x[H/4]x96) -> [ conv(96x384x8x8) --([W/2]x[H/2]x384)-> maxout(2) --([W/2]x[H/2]x192)-> max-pooling(4x4, 2) ]-> ([W/4]x[H/4]x192)
    layer3: ([W/4]x[H/4]x192) -> [ conv(192x384x5x5) --([W/4]x[H/4]x384)-> maxout(2) --([W/4]x[H/4]x192)-> max-pooling(2x2) ] -> ([W/8]x[Hx8]x192)
    layer4: ([W/8]x[H/8]x192) -> (D=[W/8]x[H/8]x192) -> fc(Dx500) --(500)-> maxout(5) --(100)-> dropout(0.5) -> softmax(10) -> result

    :param input_tensor:    The input of teacher model with shape (BatchSize, W, H, 3).
    :param trainable:       If true, train the teacher model.
    :param temp:            The temperature parameter.
    """
    with tf.variable_scope('conv_layer1'):
        conv_out_1 = tf.layers.conv2d(input_tensor, 96*2, (8,8), padding='same', trainable=trainable, name="conv")
        maxout_out_1 = maxout(conv_out_1, 96, axis=3)
        pooling_out_1 = tf.layers.max_pooling2d(maxout_out_1, 4, 2, name='max_pooling')
    with tf.variable_scope('conv_layer2'):
        #conv_out_2 = tf.layers.conv2d(dropout_out_1, 192, (3,3), padding="same", trainable=trainable, name="conv")
        conv_out_2 = tf.layers.conv2d(pooling_out_1, 192*2, (8,8), padding="same", trainable=trainable, name="conv")
        maxout_out_2 = maxout(conv_out_2, 192, axis=3)
        pooling_out_2 = tf.layers.max_pooling2d(maxout_out_2, 4, 2, name="max_pooling")
    with tf.variable_scope('conv_layer3'):
        #conv_out_3 = tf.layers.conv2d(dropout_out_2, 192, (5,5), padding="same", trainable=trainable, name="conv")
        conv_out_3 = tf.layers.conv2d(pooling_out_2, 192*2, (5,5), padding="same", trainable=trainable, name="conv")
        maxout_out_3 = maxout(conv_out_3, 192, axis=3)
        pooling_out_3 = tf.layers.max_pooling2d(maxout_out_3, 2, 2, name="max_pooling")
    with tf.variable_scope('fc_layer'):
        #tensor_shape = dropout_out_3.get_shape().as_list()
        tensor_shape = pooling_out_3.get_shape().as_list()
        new_dim = np.prod(tensor_shape[1:])
        fc_input_tensor = tf.reshape(pooling_out_3, [-1, new_dim])
        #fc_input_tensor = tf.reshape(dropout_out_3, [-1, new_dim])
        fc_out = tf.layers.dense(fc_input_tensor, 500*5, trainable=trainable, name="fc")
        maxout_out_4 = maxout(fc_out, 500, axis=1)
        dropout_out = tf.layers.dropout(maxout_out_4, rate=0.5, training=is_train)
        logits = tf.layers.dense(dropout_out, 10, trainable=trainable, name="logits") / temp
        probs = tf.nn.softmax(logits, name="probs")

    return logits, probs


def mnist_teacher_model(input_tensor, trainable=True, is_train=True, temp=1.0):
    """
    Teacher model of MNIST. This model is defined in http://arxiv.org/abs/1503.02531
    The model has two hidden layers of 1200 ReLU units over 60000 training instances.
    The model is regularized using dropout and weight-constraints. 50% dropout rate for all hidden unit
    and 20% dropout rate for visible units 
    """
    input_tensor = tf.reshape(input_tensor, shape=[-1, 28*28])
    fc_out1 = tf.layers.dense(input_tensor, units=1200, activation=tf.nn.relu, name="fc1", trainable=trainable)
    dropout_out_1 = tf.layers.dropout(fc_out1, rate=0.5, training=is_train, name="dropout1")
    fc_out2 = tf.layers.dense(dropout_out_1, units=1200, activation=tf.nn.relu, name="fc2", trainable=trainable)
    dropout_out_2 = tf.layers.dropout(fc_out2, rate=0.5, training=is_train, name="dropout2")
    fc_out3 = tf.layers.dense(dropout_out_2, units=10, name="fc3", trainable=trainable)
    dropout_out_3 = tf.layers.dropout(fc_out3, rate=0.2, training=is_train, name="dropout3")
    logits = dropout_out_3 / temp
    probs = tf.nn.softmax(logits, name="probs")
    return logits, probs

