import random
import tensorflow as tf
from collections import deque


class QNetwork:
    """
    A Q-Network implementation
    """
    def __init__(self, input_size, output_size, hidden_layers_size, gamma):
        self.q_target = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)  # enumerated actions
        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
        layer = self.states
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.layers.dense(inputs=layer, units=output_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.predictions = tf.gather_nd(self.output, indices=self.actions)
        self.labels = self.r + (gamma * tf.reduce_max(self.q_target, axis=1))
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


class ReplayMemory:
    """
    A cyclic Experience Replay memory buffer
    """
    memory: deque = None
    counter = 0

    def __init__(self, size, seed=None):
        self.memory = deque(maxlen=size)
        if seed is not None:
            random.seed(seed)

    def __len__(self):
        return len(self.memory)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n, or_less=False):
        if or_less and n > self.counter:
            n = self.counter
        return random.sample(self.memory, n)
