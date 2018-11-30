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
    samples_till_learning = 0
    last_lr = None

    previous_cost = None

    def __init__(self, name, hidden_layers_size, learning_rate, gamma, learning_batch_size, samples_till_learning, batches_to_checkpoint,
                 epsilon, tau, **kwargs):
        self.learning_batch_size = learning_batch_size
        self.batches_to_checkpoint = batches_to_checkpoint
        self.tau = tau
        self.epsilon = epsilon
        self.default_lr = learning_rate
        self.samples_till_learning = samples_till_learning
        self.session = tf.Session()
        self.qnn = dqn.QNetwork(hidden_layers_size, gamma)
        self.q_target = dqn.QNetwork(hidden_layers_size, gamma)
        self.session.run(tf.global_variables_initializer())
        super(QPlayer, self).__init__(name)

    def select_cell(self, board, **kwargs):
        available_cells = np.where(board == 0)[0]
        rnd = random.random()
        eps = self.epsilon / (10**len(str(int(kwargs['counter']))))
        if rnd < eps:
            cell = random.choice(available_cells)
            self.logger.debug("Choosing a random cell: %s [Epsilon = %s]", cell, eps)
        else:
            prediction = self.session.run(self.qnn.output,feed_dict={self.qnn.states: np.expand_dims(board, axis=0)})
            prediction = np.squeeze(prediction)
            '''
            available_cells = np.where(board == 0)[0]
            for i in range(9):
                prediction[i] = prediction[i] if i in available_cells else -np.inf
            '''
            cell = np.argmax(prediction)
            self.logger.debug("Predicting next cell - board: %s | prediction: %s | cell: %s [Epsilon = %s]", board, prediction, cell, eps)
        return cell

    @staticmethod
    def _fetch_from_batch(batch, key):
        a = np.array(list(map(lambda x: x[key], batch)))
        if len(a.shape) < 2:
            a = np.expand_dims(a, axis=1)
        return a

    def learn(self, memory, **kwargs):
        self.logger.debug('Memory counter = %s',memory.counter)
        if memory.counter % self.samples_till_learning != 0 or memory.counter < self.learning_batch_size:
            pass
        else:
            self.logger.info('Initiating learning procedure')
            batch = memory.sample(self.learning_batch_size)
            qt = self.session.run(self.q_target.output,feed_dict={self.q_target.states: self._fetch_from_batch(batch,'next_state')})
            if self.previous_cost is not None:
                lr = min(self.default_lr, self.default_lr * (10**len(str(abs(int(self.previous_cost)))))/(1000))
            else:
                lr = self.default_lr
            self.last_lr = lr
            _, cost, o, a, p, l, h, r = \
                self.session.run([self.qnn.optimizer, self.qnn.cost, self.qnn.output, self.qnn.enum_actions, self.qnn.predictions, self.qnn.labels, self.qnn.qtarget_highest_value, self.qnn.r],
                                       feed_dict={self.qnn.states: self._fetch_from_batch(batch,'state'),
                                                  self.qnn.r: self._fetch_from_batch(batch,'reward'),
                                                  self.qnn.enum_actions: np.array(list(enumerate(map(lambda x: x['action'], batch)))),
                                                  self.qnn.q_target: qt,
                                                  self.qnn.learning_rate: lr})
            self.logger.debug("Output: %s | Action: %s | GatherND (Predictions): %s | Labels: %s | Q-Target Max: %s | Reward: %s",o,a,p,l,h,r)
            self.previous_cost = cost
            self.logger.info('Q-Network cost: %s | Batch number: %s', cost, memory.counter / self.samples_till_learning)
            if memory.counter % (self.batches_to_checkpoint * self.learning_batch_size) == 0:
                self.logger.info('Copying Q-Network to Q-Target')
                tf_vars = tf.trainable_variables()
                num_of_vars = len(tf_vars)
                operations = []
                for i,v in enumerate(tf_vars[0:num_of_vars//2]):
                    operations.append(tf_vars[i+num_of_vars//2].assign((v.value()*self.tau) + ((1-self.tau)*tf_vars[i+num_of_vars//2].value())))
                self.session.run(operations)
            return cost

    def shutdown(self):
        self.session.close()

