import logging
import random
import dqn
import numpy as np
import tensorflow as tf
from abc import abstractmethod


class Player:
    name = None
    player_id = None

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


class Novice(Player):
    def find_two_of_three(self, board, which_player_id):
        cell = None
        winning_options = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                           [0, 3, 6], [1, 4, 7], [2, 5, 8],
                           [0, 4, 8], [2, 4, 6]]
        random.shuffle(winning_options)
        for seq in winning_options:
            s = board[seq[0]] + board[seq[1]] + board[seq[2]]
            if s == 2 * which_player_id:
                a = np.array([board[seq[0]], board[seq[1]], board[seq[2]]])
                c = np.where(a == 0)[0][0]
                cell = seq[c]
                break
        return cell

    def select_cell(self, board, **kwargs):
        cell = self.find_two_of_three(board,self.player_id)
        if cell is None:
            cell = self.find_two_of_three(board,-self.player_id)
        if cell is None:
            available_cells = np.where(board == 0)[0]
            cell = random.choice(available_cells)
        return cell

    def learn(self, memory: dqn.ReplayMemory, **kwargs):
        pass


class QPlayer(Player):
    qnn: dqn.QNetwork = None
    q_target: dqn.QNetwork = None
    session: tf.Session = None
    train_counter = 0
    counter = 0

    def __init__(self, name, hidden_layers_size, learning_rate, gamma, learning_batch_size,
                 batches_to_q_target_switch, epsilon, tau, pre_train_steps=0):
        self.learning_batch_size = learning_batch_size
        self.batches_to_q_target_switch = batches_to_q_target_switch
        self.tau = tau
        self.train_counter = 0
        self.pre_train_steps = pre_train_steps
        self.counter = 0
        self.init_epsilon = epsilon
        self.init_lr = learning_rate
        self.session = tf.Session()
        self.qnn = dqn.QNetwork(9, 9, hidden_layers_size, gamma)
        self.q_target = dqn.QNetwork(9, 9, hidden_layers_size, gamma)
        self.session.run(tf.global_variables_initializer())
        super(QPlayer, self).__init__(name)

    def select_cell(self, board, **kwargs):
        available_cells = np.where(board == 0)[0]
        rnd = random.random()
        eps = self.init_epsilon
        self.counter += 1
        if self.counter <= self.pre_train_steps or rnd < eps:
            cell = random.choice(available_cells)
            logging.debug("Choosing a random cell: %s [Epsilon = %s]", cell, eps)
        else:
            prediction = self.session.run(self.qnn.output,feed_dict={self.qnn.states: np.expand_dims(board, axis=0)})
            prediction = np.squeeze(prediction)
            #print(prediction)
            nth = int(kwargs.get('nth',1))
            if nth == 1:
                cell = np.argmax(prediction)
            else:
                cell = (-prediction).argsort()[nth-1]
            logging.debug("Predicting next cell - board: %s | prediction: %s | cell: %s [Epsilon = %s]", board, prediction, cell, eps)
        return cell

    @staticmethod
    def _fetch_from_batch(batch, key, enum=False):
        if enum:
            return np.array(list(enumerate(map(lambda x: x[key], batch))))
        else:
            return np.array(list(map(lambda x: x[key], batch)))

    def learn(self, memory: dqn.ReplayMemory, **kwargs):
        logging.debug('Memory counter = %s',memory.counter)
        if memory.counter % self.learning_batch_size != 0 or memory.counter < self.learning_batch_size:
            pass
        else:
            logging.debug('Starting learning procedure')
            batch = memory.sample(self.learning_batch_size)
            qt = self.session.run(self.q_target.output,feed_dict={self.q_target.states: self._fetch_from_batch(batch,'next_state')})
            terminals = self._fetch_from_batch(batch,'game_over')
            for i in range(terminals.size):
                if terminals[i]:
                    qt[i] = np.zeros(9)
            lr = self.init_lr
            self.train_counter += 1
            _, cost = self.session.run([self.qnn.optimizer, self.qnn.cost],
                                       feed_dict={self.qnn.states: self._fetch_from_batch(batch,'state'),
                                                  self.qnn.r: self._fetch_from_batch(batch,'reward'),
                                                  self.qnn.actions: self._fetch_from_batch(batch, 'action', enum=True),
                                                  self.qnn.q_target: qt,
                                                  self.qnn.learning_rate: lr})
            logging.info('Batch number: %s | Q-Network cost: %s | Learning rate: %s',
                         self.train_counter, cost, lr)
            if memory.counter % (self.batches_to_q_target_switch * self.learning_batch_size) == 0:
                logging.info('Copying Q-Network to Q-Target')
                tf_vars = tf.trainable_variables()
                num_of_vars = len(tf_vars)
                operations = []
                for i,v in enumerate(tf_vars[0:num_of_vars//2]):
                    operations.append(tf_vars[i+num_of_vars//2].assign((v.value()*self.tau) + ((1-self.tau)*tf_vars[i+num_of_vars//2].value())))
                self.session.run(operations)
            return cost

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.session, filename)

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.session, filename)

    def shutdown(self):
        self.session.close()

