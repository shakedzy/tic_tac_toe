import random
import tensorflow as tf
from collections import deque


class QNetwork:
    def __init__(self, hidden_layers_size, gamma):
        self.q_target = tf.placeholder(shape=(None,9), dtype=tf.float32)
        self.r = tf.placeholder(shape=None,dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, 9), dtype=tf.float32)
        self.enum_actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
        layer = self.states
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.layers.dense(inputs=layer, units=9,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.predictions = tf.gather_nd(self.output, indices=self.enum_actions)
        self.qtarget_highest_value = tf.reduce_max(self.q_target)
        self.labels = self.r + (gamma * self.qtarget_highest_value)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


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
