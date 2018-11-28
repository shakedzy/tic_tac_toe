import logging
import random
import dqn
import numpy as np
import tensorflow as tf
from abc import abstractmethod


class Player:
    name = None

    def __init__(self, name):
        self.name = name

    def shutdown(self):
        pass

    @abstractmethod
    def select_cell(self, board, **kwargs):
        pass

    @abstractmethod
    def learn(self, memory: dqn.ReplayMemory, **kwargs):
        pass


class Human(Player):
    def select_cell(self, board, **kwargs):
        cell = input("Select cell to fill:\n678\n345\n012\ncell number: ")
        return cell

    def learn(self, memory, **kwargs):
        pass


class Drunk(Player):
    def select_cell(self, board, **kwargs):
        available_cells = np.where(board == 0)[0]
        return random.choice(available_cells)

    def learn(self, memory: dqn.ReplayMemory, **kwargs):
        pass


class QPlayer(Player):
    logger = logging.getLogger("logger")
    qnn: dqn.QNetwork = None
    q_target: dqn.QNetwork = None
    session: tf.Session = None

    learning_batch_size = 0
    batches_to_checkpoint = 0

    def __init__(self, name, hidden_layers_size, learning_rate, gamma, learning_batch_size, batches_to_checkpoint,
                 epsilon=0.01, tau=1, **kwargs):
        self.learning_batch_size = learning_batch_size
        self.batches_to_checkpoint = batches_to_checkpoint
        self.tau = tau
        self.epsilon = epsilon
        self.session = tf.Session()
        self.qnn = dqn.QNetwork(hidden_layers_size, gamma, learning_rate)
        self.q_target = dqn.QNetwork(hidden_layers_size, gamma, learning_rate)
        self.session.run(tf.global_variables_initializer())
        super(QPlayer, self).__init__(name)

    def select_cell(self, board, **kwargs):
        e = random.random()
        if e < self.epsilon:
            cell = random.randint(0,8)
            self.logger.debug("Choosing a random cell: %s", cell)
        else:
            prediction = self.session.run(self.qnn.output,feed_dict={self.qnn.states: np.expand_dims(board, axis=0)})
            prediction = np.squeeze(prediction)
            cell = np.argmax(prediction)
            self.logger.debug("Predicting next cell - board: %s | prediction: %s | cell: %s", board, prediction, cell)
        return cell

    @staticmethod
    def _fetch_from_batch(batch, key):
        a = np.array(list(map(lambda x: x[key], batch)))
        if len(a.shape) < 2:
            a = np.expand_dims(a, axis=1)
        return a

    def learn(self, memory, **kwargs):
        self.logger.debug('Memory counter = %s',memory.counter)
        if memory.counter % self.learning_batch_size != 0:
            pass
        else:
            self.logger.info('Initiating learning procedure')
            batch = memory.sample(self.learning_batch_size)
            qt = self.session.run(self.q_target.output,feed_dict={self.q_target.states: self._fetch_from_batch(batch,'next_state')})
            _, cost = self.session.run([self.qnn.optimizer, self.qnn.cost], feed_dict={self.qnn.states: self._fetch_from_batch(batch,'state'),
                                                                                       self.qnn.r: self._fetch_from_batch(batch,'reward'),
                                                                                       self.qnn.actions: self._fetch_from_batch(batch, 'action'),
                                                                                       self.qnn.q_target: qt})
            self.logger.info('Q-Network cost: %s | Batch number: %s', cost, memory.counter / self.learning_batch_size)
            if memory.counter % (self.batches_to_checkpoint * self.learning_batch_size) == 0:
                self.logger.info('Copying Q-Network to Q-Target')
                tf_vars = tf.trainable_variables()
                num_of_vars = len(tf_vars)
                operations = []
                for i,v in enumerate(tf_vars[0:num_of_vars//2]):
                    operations.append(tf_vars[i+num_of_vars//2].assign((v.value()*self.tau) + ((1-self.tau)*tf_vars[i+num_of_vars//2].value())))
                self.session.run(operations)

    def shutdown(self):
        self.session.close()

