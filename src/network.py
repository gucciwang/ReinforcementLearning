from __future__ import division
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
from utilities import *

BETA = 0.01
NORM_CLIP = 40.0
SMALL_VALUE = 1e-20

class A3C_Network():
    def __init__(self, scope, a_size, trainer):
        self.scope = scope
        with tf.variable_scope(scope):
            # Input layer
            #self.inputs = tf.placeholder(shape=[None, 80, 80, 1], dtype = tf.float32,name='inputs')
            self.inputs = tf.placeholder("float", [None, 80, 80, 1], name="inputs")
            #self.imageIn = tf.reshape(self.inputs, shape=[-1,80,80,1])

            # Layer 1
            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs,
                filters = 32,
                kernel_size = [5,5],
                padding = "same")
            self.pool1 = tf.layers.max_pooling2d(inputs = self.conv1,
                pool_size = [2,2], strides = 2)

            # Layer 2
            self.conv2 = tf.layers.conv2d(
                inputs = self.pool1,
                filters = 32,
                kernel_size = [5,5],
                padding = "same",
                activation = tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(inputs = self.conv2,
                pool_size = [2,2], strides = 2)

            # Layer 3
            self.conv3 = tf.layers.conv2d(
                inputs = self.pool2,
                filters = 64,
                kernel_size = [4,4],
                padding = "same",
                activation = tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(inputs = self.conv3,
                pool_size = [2,2], strides = 2)

            # Layer 4
            self.conv4 = tf.layers.conv2d(
                inputs = self.pool3,
                filters = 64,
                kernel_size = [3,3],
                padding = "same",
                activation = tf.nn.relu)
            self.pool4 = tf.layers.max_pooling2d(inputs = self.conv4,
                pool_size = [2,2], strides = 2)

            #self.hidden = slim.fully_connected(slim.flatten(self.pool4),512,activation_fn=None)
            h = slim.flatten(self.pool4)
            h = layer.fully_connected(h, 512, activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.0))
            lstm_input = tf.expand_dims(h, [0])
            step_size = tf.shape(self.inputs)[:1]

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(512)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.lstm_state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = [c_in, h_in]
            lstm_state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_out, lstm_state_out = tf.nn.dynamic_rnn(	lstm_cell, lstm_input,
															initial_state=lstm_state_in,
			 												sequence_length=step_size)

            lstm_c, lstm_h = lstm_state_out
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            lstm_out = tf.reshape(lstm_out, [-1, 512])
            self.policy = layer.fully_connected(lstm_out, a_size, activation_fn=tf.nn.softmax,
												weights_initializer=normalized_columns_initializer(0.01),
               									biases_initializer=tf.constant_initializer(0))
            self.value = layer.fully_connected(	lstm_out, 1, activation_fn=None,
												weights_initializer=normalized_columns_initializer(1.0),
               									biases_initializer=tf.constant_initializer(0))

            '''
            # Build LSTM for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(512)

            # Following parameters called during training
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.lstm_state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            # Build LSTM parameters
            rnn_in = tf.expand_dims(self.hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 512])

            # Actor (Policy) and Critic (Value) Output layers
            self.policy = slim.fully_connected(rnn_out, a_size, activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            '''

            # If worker agent, implement gradient descent
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32, name="discounted_reward")
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32, name="advantage")

                v = tf.reshape(self.value, [-1])
                log_policy = tf.log(tf.clip_by_value(self.policy, SMALL_VALUE, 1.0))
                responsible_outputs = tf.reduce_sum(tf.multiply(log_policy, self.actions_onehot), reduction_indices=1)

                # Loss functions
                policy_loss = -tf.reduce_sum(responsible_outputs*self.advantages)
                value_loss = 0.5 * tf.reduce_sum(tf.square(v - self.target_v))
                entropy = - tf.reduce_sum(self.policy * log_policy)
                self.total_loss = 0.5 * value_loss + policy_loss - entropy * BETA

                # Get gradients from local network using local losses
                local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                gradients = tf.gradients(self.total_loss, local_params)
                grads, grad_norms = tf.clip_by_global_norm(gradients, NORM_CLIP)

                # Apply local gradients to global network
                global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(list(zip(grads, global_params)))

    def update_network_op(self, from_scope):
    	with tf.variable_scope(self.scope):
    		to_scope = self.scope
    		from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    		to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    		ops = []
    		for from_var,to_var in zip(from_vars,to_vars):
    			ops.append(to_var.assign(from_var))

    		return tf.group(*ops)
