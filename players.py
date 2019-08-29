import random
import logging
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from dqn import DeepQNetworkModel
from memory_buffers import ExperienceReplayMemory


class Player:
    """
    Base class for all player types
    """
    name = None
    player_id = None

    def __init__(self):
        pass

    def shutdown(self):
        pass

    def add_to_memory(self, add_this):
        pass

    def save(self, filename):
        pass

    @abstractmethod
    def select_cell(self, board, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass


class Human(Player):
    """
    This player type allow a human player to play the game
    """
    def select_cell(self, board, **kwargs):
        cell = input("Select cell to fill:\n678\n345\n012\ncell number: ")
        return cell

    def learn(self, **kwargs):
        pass


class Drunk(Player):
    """
    Drunk player always selects a random valid move
    """
    def select_cell(self, board, **kwargs):
        available_cells = np.where(board == 0)[0]
        return random.choice(available_cells)

    def learn(self, **kwargs):
        pass


class Novice(Player):
    """
    A more sophisticated bot, which follows the following strategy:
    1) If it already has 2-in-a-row, capture the required cell for 3
    2) If not, and if the opponent has 2-in-a-row, capture the required cell to prevent hi, from winning
    3) Else, select a random vacant cell
    """
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

    def learn(self, **kwargs):
        pass


class QPlayer(Player):
    """
    A reinforcement learning agent, based on Double Deep Q Network model
    This class holds two Q-Networks: `qnn` is the learning network, `q_target` is the semi-constant network
    """
    def __init__(self, session, hidden_layers_size, gamma, learning_batch_size, batches_to_q_target_switch, tau, memory_size,
                 maximize_entropy=False, var_scope_name=None):
        """
        :param session: a tf.Session instance
        :param hidden_layers_size: an array of integers, specifying the number of layers of the network and their size
        :param gamma: the Q-Learning discount factor
        :param learning_batch_size: training batch size
        :param batches_to_q_target_switch: after how many batches (trainings) should the Q-network be copied to Q-Target
        :param tau: a number between 0 and 1, determining how to combine the network and Q-Target when copying is performed
        :param memory_size: size of the memory buffer used to keep the training set
        :param maximize_entropy: boolean, should the network try to maximize entropy over direct future rewards
        :param var_scope_name: the variable scope to use for the player
        """
        layers_size = [item for sublist in [[9],hidden_layers_size,[9]] for item in sublist]
        self.session = session
        self.model = DeepQNetworkModel(session=self.session, layers_size=layers_size,
                                       memory=ExperienceReplayMemory(memory_size),default_batch_size=learning_batch_size,
                                       gamma=gamma, double_dqn=True,
                                       learning_procedures_to_q_target_switch=batches_to_q_target_switch,
                                       tau=tau, maximize_entropy=maximize_entropy, var_scope_name=var_scope_name)
        self.session.run(tf.global_variables_initializer())
        super(QPlayer, self).__init__()

    def select_cell(self, board, **kwargs):
        return self.model.act(board, epsilon=kwargs['epsilon'])

    def learn(self, **kwargs):
        return self.model.learn(learning_rate=kwargs['learning_rate'])

    def add_to_memory(self, add_this):
        state = self.player_id * add_this['state']
        next_state = self.player_id * add_this['next_state']
        self.model.add_to_memory(state=state, action=add_this['action'], reward=add_this['reward'],
                                 next_state=next_state, is_terminal_state=add_this['game_over'])

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.session, filename)

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.session, filename)

    def shutdown(self):
        try:
            self.session.close()
        except Exception as e:
            logging.warning('Failed to close session', e)
