import random
import tensorflow as tf
from collections import deque


class QNetwork:
    graph = None
    saver = None
    checkpoint_file_name = ''

    output = None
    cost = None
    optimizer = None

    def __init__(self, hidden_layers_size, gamma, learning_rate,
                 checkpoint_file_name='dqn.ckpt'):
        self.graph = tf.Graph()
        self.checkpoint_file_name = checkpoint_file_name
        with self.graph.as_default():
            with tf.Session() as sess:
                q_target = tf.placeholder(name='q_target',shape=(None,9), dtype=tf.float32)
                r = tf.placeholder(name='r',shape=(None,1),dtype=tf.float32)
                states = tf.placeholder(name='states', shape=(None, 9), dtype=tf.float32)
                actions = tf.placeholder(name='actions', shape=None, dtype=tf.int32)
                layer = states
                for l in hidden_layers_size:
                    layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu)
                self.output = tf.layers.dense(inputs=layer, units=9, name='output')
                self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=r+(gamma*tf.reduce_max(q_target)),
                                                                        predictions=tf.gather(self.output,
                                                                                              indices=actions,
                                                                                              axis=1)))
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
                self.saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                #print([n.name for n in tf.get_default_graph().as_graph_def().node])
                #print('XX',tf.trainable_variables())

    def predict(self, current_states):
        with self.graph.as_default():
            with tf.Session() as sess:
                #print('HERE',[n.name for n in tf.get_default_graph().as_graph_def().node if '/' not in n.name])
                predictions = sess.run(self.output,feed_dict={states:current_states})
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
