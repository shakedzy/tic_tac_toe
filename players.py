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

    learning_batch_size = 0
    batches_to_checkpoint = 0

    def __init__(self, name, hidden_layers_size, learning_rate, gamma, learning_batch_size, batches_to_checkpoint,
                 **kwargs):
        self.qnn = dqn.QNetwork(hidden_layers_size, gamma, learning_rate)
        self.q_target = dqn.QNetwork(hidden_layers_size, gamma, learning_rate)
        self.learning_batch_size = learning_batch_size
        self.batches_to_checkpoint = batches_to_checkpoint
        super(QPlayer, self).__init__(name)

    def select_cell(self, board, **kwargs):
        return self.qnn.predict(board)

    def learn(self, memory, **kwargs):
        self.logger.debug('Memory counter = %s',memory.counter)
        if memory.counter % self.learning_batch_size != 0:
            pass
        else:
            self.logger.info('Initiating learning procedure')
            batch = memory.sample(self.learning_batch_size)
            qt = self.q_target.predict(np.array(list(map(lambda x: x['next_state'], batch))))
            cost = self.qnn.train(states=np.array(list(map(lambda x: x['state'], batch))),
                                  r=np.array(list(map(lambda x: x['reward'], batch))),
                                  q_target=qt)
            self.logger.info('Q-Network cost: %s | Batch number: %s', cost, memory.counter / self.learning_batch_size)
            if memory.counter % (self.batches_to_checkpoint * self.learning_batch_size) == 0:
                self.logger.info('Copying Q-Network to Q-Target')
                self.qnn.save_graph()
                self.q_target.load_graph()

