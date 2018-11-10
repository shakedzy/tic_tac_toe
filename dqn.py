import random
import tensorflow as tf
from collections import deque


class QNetwork:
    session = None
    graph = None
    saver = None
    checkpoint_file_name = ''

    output = None
    cost = None
    optimizer = None

    def __init__(self, session, saver, gamma=0.8, hidden_layers_size=[20,20], learning_rate=0.0003,
                 checkpoint_file_name='dqn.ckpt'):
        self.session = session
        self.saver = saver
        self.graph = tf.Graph()
        self.checkpoint_file_name = checkpoint_file_name
        with self.graph.as_default():
            q_target = tf.placeholder(name='q_target',shape=(None,9), dtype=tf.float32)
            r = tf.placeholder(name='r',shape=(None,1),dtype=tf.float32)
            states = tf.placeholder(name='states', shape=(None, 9), dtype=tf.int8)
            layer = states
            for l in hidden_layers_size:
                layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu)
            self.output = tf.layers.dense(inputs=layer, units=9)
            self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=r+(gamma*tf.reduce_max(q_target)),
                                                                    predictions=self.output))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        with self.graph.as_default():
            predictions = self.session.run(self.output,feed_dict={states:states})
        return predictions

    def train(self, states, r, q_target):
        with self.graph.as_default():
            _, cost = self.session.run([self.optimizer, self.cost], feed_dict={states:states, r:r, q_target:q_target})
        return cost

    def save_graph(self):
        with self.graph.as_default():
            self.saver.save(self.session, self.checkpoint_file_name)

    def load_graph(self):
        with self.graph.as_default():
            self.saver.restore(self.session, self.checkpoint_file_name)


class ReplayMemory:
    memory = None

    def __init__(self, size, seed=None):
        self.memory = deque(maxlen=size)
        if seed is not None:
            random.seed(seed)

    def sample(self, n):
        return random.sample(self.memory, n)
