import random
import tensorflow as tf
from collections import deque


class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate):
        self.q_target = tf.placeholder(shape=(None,9), dtype=tf.float32)
        self.r = tf.placeholder(shape=(None,1),dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, 9), dtype=tf.float32)
        self.actions = tf.placeholder(shape=None, dtype=tf.int32)
        layer = self.states
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu)
        self.output = tf.layers.dense(inputs=layer, units=9)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.r+(gamma*tf.reduce_max(self.q_target)),
                                                                predictions=tf.gather(self.output,
                                                                                      indices=self.actions,
                                                                                      axis=1)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)


class ReplayMemory:
    # Single memory element: dict(state,action,reward,next_state)
    memory: deque = None
    counter = 0

    def __init__(self, size, seed=None):
        self.memory = deque(maxlen=size)
        if seed is not None:
            random.seed(seed)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        return random.sample(self.memory, n)
