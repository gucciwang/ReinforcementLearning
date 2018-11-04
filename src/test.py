import tensorflow as tf
import numpy as np
import scipy.misc
import os
import random
from functools import partial
from network import A3C_Network
from utilities import *
from datetime import datetime
import time
import tensorflow.contrib.slim as slim
import gym
import scipy.signal
import json
import sys
import argparse
from random import choice

#Parse Inputs - Default parameters listed below
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, help='learning rate', default = 0.0001)
parser.add_argument('--gamma', type=float, help='discount factor for rewards', default = 0.99)
parser.add_argument('--workers', type=int, help='number of training processes', default = 32)
parser.add_argument('--num-steps', type=int, help='number of forward steps in A3C', default = 20)
parser.add_argument('--max-episode-length', type=int, help='maximum length of an episode', default = 6000)
parser.add_argument('--env',help='environment to train on', default = 'MsPacman-v0')
parser.add_argument('--a-size', type=int, help='number of actions available in MsPacman env', default = 9)
parser.add_argument('--sleep-time', type=float, help='time between each test (seconds)', default = 15)
parser.add_argument('--render', help='render game', default = True)
parser.add_argument('--skip-rate', type=int, help='frame skip rate', default = 4)
parser.add_argument('--load-model', help='load existing model', default = True)
parser.add_argument('--model-path', help='model path', default = './model')
args = parser.parse_args()

#Each worker is an independent thread, and updates the global network.
class A3C_Test_Worker():
	def __init__(self, number, trainer, global_episodes, args):
		self.name = "agent_" + str(number)
		self.number = number
		self.start_time = args.start_time
		self.model_path = args.model_path
		self.discount_factor = args.gamma
		self.num_steps = args.num_steps
		self.a_size = args.a_size
		self.sleep_time = args.sleep_time
		self.global_episodes = global_episodes	# Number of global episodes

		#Setup Atari Environment
		self.env = atari_env(args.env, {'crop1': 2, 'crop2': 10, 'dimension2': 84}, args) # crop and resize info for MsPacman

		#Create and initialize local network with global network parameters
		self.local_net = A3C_Network(self.name, self.a_size, trainer)
		self.update_local_ops = self.local_net.update_network_op('global')

	def test(self, sess, saver):
		print('Test Agent')

		tf.set_random_seed(1)
		reward_sum = 0
		start_time = time.time()
		num_tests = 0
		reward_total_sum = 0
		last_lstm_state = self.local_net.lstm_state_init
		sess.run(self.update_local_ops) #Initialzie weights to global network
		state = self.env.reset()

		max_score = 0
		while True:

			policy, last_lstm_state = sess.run([self.local_net.policy, self.local_net.state_out],
													feed_dict={	self.local_net.inputs:[state],
																self.local_net.state_in[0]:last_lstm_state[0],
																self.local_net.state_in[1]:last_lstm_state[1]})

			action = np.argmax(policy[0]) #pick action with highest probability
			state, reward, done, info = self.env.step(action) #Pick action

			reward_sum += reward
			self.env.render()

			if done and not info:
				state = self.env.reset()
				last_lstm_state = self.local_net.lstm_state_init
			elif info:
				num_tests += 1
				reward_total_sum += reward_sum
				reward_mean = reward_total_sum / num_tests
				string_reward_mean = "{0:.4f}".format(reward_mean)

				#Print episode summary
				stop_time = time.time()
				time_elapsed = str(time.strftime("%H:%M:%S", time.gmtime(stop_time-start_time)))
				summary = str(datetime.now())[:-7]+" |  Elapsed: "+time_elapsed+"  |  Count: "+str(num_tests)+"  |  Episode Reward: "+str(reward_sum)+"  |  Average Reward: "+string_reward_mean
				print(summary)

				reward_sum = 0
				state = self.env.reset()
				last_lstm_state = self.local_net.lstm_state_init

if __name__ == '__main__':
	tf.reset_default_graph()

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	with tf.device("/cpu:0"):
		global_network = A3C_Network('global', args.a_size, None) # Generate global network
		global_episodes = tf.Variable(0,dtype=tf.int32, name='global_episodes', trainable=False)
		trainer = tf.train.AdamOptimizer(learning_rate=args.lr) #Use Adam Optimizer
		workers = []
		args.start_time = time.time()

	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=5)
		if args.load_model == True:
			print ('Loading Graph...')
			ckpt = tf.train.get_checkpoint_state(args.model_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
			print ('Graph Loaded.')
		else:
			sess.run(tf.global_variables_initializer())

		A3C_Render = A3C_Test_Worker(1, trainer, global_episodes, args)
		A3C_Render.test(sess, saver)
