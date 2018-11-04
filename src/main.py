import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import scipy.signal
import os
import threading
import json
import sys
import multiprocessing
from utilities import atari_env
from network import A3C_Network
from agent import A3C_Worker
import numpy as np
import matplotlib.pyplot as plt
import argparse
from random import choice
import time

#Parse Inputs - Default parameters listed below
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, help='learning rate', default = 0.0001)
parser.add_argument('--gamma', type=float, help='discount factor for rewards', default = 0.99)
parser.add_argument('--workers', type=int, help='number of training processes', default = 16)
parser.add_argument('--num-steps', type=int, help='number of forward steps in A3C', default = 20)
parser.add_argument('--max-episode-length', type=int, help='maximum length of an episode', default = 6000)
parser.add_argument('--env',help='environment to train on', default = 'MsPacman-v0')
parser.add_argument('--a-size', type=int, help='number of actions available in MsPacman env', default = 9)
parser.add_argument('--render', help='render game', default = False)
parser.add_argument('--sleep-time', type=float, help='time between each test (seconds)', default = 7)
parser.add_argument('--skip-rate', type=int, help='frame skip rate', default = 4)
parser.add_argument('--load', help='load existing model', default = False)
parser.add_argument('--model-path', help='model path', default = '../models/model')
args = parser.parse_args()

if __name__ == '__main__':
    tf.reset_default_graph()

    #Create directory to save model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with tf.device("/cpu:0"):
        #Initialize global network
        global_network = A3C_Network('global', args.a_size, None) # Generate global network
        global_episodes = tf.Variable(0,dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=args.lr) #Use Adam Optimizer
        workers = []
        args.start_time = time.time()

        # Create worker threads. Note: Last index contains test agent
        for i in range(0, args.workers+1):
            workers.append(A3C_Worker(i, trainer, global_episodes, args))

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        if args.load:
            print ('Loading Graph...')
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print ('Graph Loaded.')
        else:
            sess.run(tf.global_variables_initializer())

        thread_manager = tf.train.Coordinator() #Coordinate asynchronous threads

        #Begin asynchronous training
        worker_threads = []
        for i in range(args.workers+1):
            if i < args.workers:
                # t = threading.Thread(target=workers[i].train, args=(sess, thread_manager, saver))
                t = threading.Thread(target=workers[i].train(sess, thread_manager))
            else:
                # t = threading.Thread(target=workers[i].test, args=(sess, thread_manager, saver, args))
                t = threading.Thread(target=workers[i].test(sess, thread_manager, saver, args))
            t.start()
            time.sleep(0.5)
            worker_threads.append(t)
        thread_manager.join(worker_threads)
