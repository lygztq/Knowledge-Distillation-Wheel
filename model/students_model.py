import tensorflow as tf
import numpy as np

def mnist_student_model(input_tensor, num_hidden_unit=800, is_train=True):
    """
    Student model of MNIST. This model is defined in http://arxiv.org/abs/1503.02531
    The model has two hidden layers of (n<1200) ReLU units over 60000 training instances.
    The model is regularized using dropout and weight-constraints. 50% dropout rate for all hidden unit
    and 20% dropout rate for visible units 
    """
    input_tensor = tf.reshape(input_tensor, shape=[-1, 28*28])
    fc_out1 = tf.layers.dense(input_tensor, units=num_hidden_unit, activation=tf.nn.relu, name="fc1")
    dropout_out_1 = tf.layers.dropout(fc_out1, rate=0.5, training=is_train, name="dropout1")
    fc_out2 = tf.layers.dense(dropout_out_1, units=num_hidden_unit, activation=tf.nn.relu, name="fc2")
    dropout_out_2 = tf.layers.dropout(fc_out2, rate=0.5, training=is_train, name="dropout2")
    fc_out3 = tf.layers.dense(dropout_out_2, units=10, name="fc3")
    dropout_out_3 = tf.layers.dropout(fc_out3, rate=0.2, training=is_train, name="dropout3")
    logits = dropout_out_3
    probs = tf.nn.softmax(logits, name="probs")
    return logits, probs
    
