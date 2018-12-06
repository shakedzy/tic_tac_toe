import logging, os
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import players, dqn


class Game:
    WINNING_REWARD = 1
    LOSING_REWARD = -10
    INVALID_REWARD = -100

    board = np.zeros(9)
    current_player = 1
    player1: players.Player = None
    player2: players.Player = None

    _invalid_move_played = False

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.player1.player_id = 1
        self.player2.player_id = -1
        self.reset()

    def reset(self):
        self.board = np.zeros(9)
        self.current_player = 1
        self._invalid_move_played = False

    def active_player(self):
        if self.current_player == 1:
            return self.player1
        else:
            return self.player2

    def inactive_player(self):
        if self.current_player == -1:
            return self.player1
        else:
            return self.player2

    def play(self, cell):
        self._invalid_move_played = False
        if self.board[cell] != 0:
            self._invalid_move_played = True
            return {'winner': 0,
                    'game_over': False,
                    'invalid_move': True}
        else:
            self.board[cell] = self.current_player
        status = self.game_status()
        return {'winner': status['winner'],
                'game_over': status['game_over'],
                'invalid_move': False}

    def next_player(self):
        if not self._invalid_move_played:
            self.current_player *= -1

    def game_status(self):
        winner = 0
        winning_seq = []
        winning_options = [[0,1,2],[3,4,5],[6,7,8],
                           [0,3,6],[1,4,7],[2,5,8],
                           [0,4,8],[2,4,6]]
        for seq in winning_options:
            s = self.board[seq[0]] + self.board[seq[1]] + self.board[seq[2]]
            if abs(s) == 3:
                winner = s/3
                winning_seq = seq
                break
        game_over = winner != 0 or len(list(filter(lambda z: z==0, self.board))) == 0
        return {'game_over': game_over, 'winner': winner,
                'winning_seq': winning_seq, 'board': self.board}

    def plot_board(self):
        def vector_element_to_board_cell(el):
            r = 2 * (el / 3) - 2
            c = 2 * (el % 3) - 2
            return r, c
        fig, ax = plt.subplots(figsize=(7,7))
        plt.grid(False)
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.axis('off')
        plt.plot([-1,-1],[-3,3],'m')
        plt.plot([1,1],[-3,3],'m')
        plt.plot([-3,3],[-1,-1],'m')
        plt.plot([-3,3],[1,1],'m')
        x = {'row':[], 'col': []}
        o = {'row':[], 'col': []}
        for i,p in enumerate(self.board):
            if p != 0:
                r,c = vector_element_to_board_cell(i)
                if p > 0:
                    x['row'].append(r)
                    x['col'].append(c)
                else:
                    o['row'].append(r)
                    o['col'].append(c)
        plt.plot(x['col'],x['row'],'bx',ms=100)
        plt.scatter(o['col'],o['row'],s=10000,facecolors='none',edgecolors='r')
        status = self.game_status()
        if status['winner'] != 0:
            seq = status['winning_seq']
            sr = []
            sc = []
            for element in seq:
                r,c = vector_element_to_board_cell(element)
                sr.append(r)
                sc.append(c)
                plt.plot(sc,sr,'y-',lw=7)
        plt.show()

    def print_board(self):
        row = ' '
        status = self.game_status()
        for i in reversed(range(9)):
            if self.board[i] == 1:
                cell = 'x'
            elif self.board[i] == -1:
                cell = 'o'
            else:
                cell = ' '
            if status['winner'] != 0 and i in status['winning_seq']:
                cell = cell.upper()
            row += cell + ' '
            if i % 3 != 0:
                row += '| '
            else:
                row = row[::-1]
                if i != 0:
                    row += ' \n-----------'
                print(row)
                row = ' '


def train():
    y = []
    r1 = []
    r2 = []
    random.seed(int(time()*1000))
    tf.reset_default_graph()
    logging.basicConfig(level=logging.WARN, format='%(message)s')
    memory = dqn.ReplayMemory(100000)
    p1 = players.QPlayer('Q', [80,80],
                         learning_batch_size=100, batches_to_q_target_switch=1000,
                         epsilon=0.01, pre_train_steps=0, gamma=0.9, tau=0.9,
                         learning_rate=0.0003)
    p2 = players.Novice('N')
    game = Game(p1,p2)
    total_reward_1 = 0
    total_reward_2 = 0
    for g in range(1,1000001):
        game.reset()
        #print('STARTING NEW GAME (#{})\n-------------'.format(g))
        while not game.game_status()['game_over']:
            reward1 = 0
            reward2 = 0
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.active_player().name))
            state = game.current_player * game.board
            action = int(game.active_player().select_cell(state)) if np.count_nonzero(game.board) > 0 else random.randint(0,8)
            play_status = game.play(action)
            game_over = play_status['game_over']
            next_state = game.current_player * game.board if not game_over else np.zeros(9)
            if play_status['invalid_move']:
                memory.append({'state': state, 'action': action,
                               'reward': game.INVALID_REWARD, 'next_state': next_state,
                               'game_over': game_over})
                if game.current_player == 1:
                    reward1 = game.INVALID_REWARD
                else:
                    reward2 = game.INVALID_REWARD
            elif not game_over:
                memory.append({'state': state, 'action': action,
                               'reward': 0, 'next_state': next_state,
                               'game_over': game_over})
                memory.append({'state': -state, 'action': action,
                               'reward': 0, 'next_state': -next_state,
                               'game_over': game_over})
            else:
                if play_status['winner'] == game.current_player:
                    memory.append({'state': state, 'action': action,
                                   'reward': game.WINNING_REWARD, 'next_state': next_state,
                                   'game_over': game_over})
                    memory.append({'state': -state, 'action': action,
                                   'reward': game.LOSING_REWARD, 'next_state': -next_state,
                                   'game_over': game_over})
                    if game.current_player == 1:
                        reward1 = game.WINNING_REWARD
                        reward2 = game.LOSING_REWARD
                    else:
                        reward1 = game.LOSING_REWARD
                        reward2 = game.WINNING_REWARD
                else:
                    memory.append({'state': state, 'action': action,
                                   'reward': game.LOSING_REWARD, 'next_state': next_state,
                                   'game_over': game_over})
                    memory.append({'state': -state, 'action': action,
                                   'reward': game.WINNING_REWARD, 'next_state': -next_state,
                                   'game_over': game_over})
                    if game.current_player == -1:
                        reward1 = game.WINNING_REWARD
                        reward2 = game.LOSING_REWARD
                    else:
                        reward1 = game.LOSING_REWARD
                        reward2 = game.WINNING_REWARD
            total_reward_1 += reward1
            total_reward_2 += reward2
            cost = game.active_player().learn(memory)
            if cost is not None:
                y.append(cost)
            if not game_over:
                game.next_player()
        if g % 100 == 0:
            print('Game: {g} | Average Rewards - P1: {r1}, P2: {r2}'.format(g=g,r1=total_reward_1/100.0,
                                                                            r2=total_reward_2/100.0))
            r1.append(total_reward_1 / 100.0)
            r2.append(total_reward_2 / 100.0)
            total_reward_1 = 0
            total_reward_2 = 0
        #print('-------------\nGAME OVER!')
        #game.print_board()
        #print(game.game_status())
        #print('-------------')
    p1.save(os.getcwd() + '/qx.ckpt')
    for pp in [p1,p2]:
        pp.shutdown()
    with open('costs','w') as f:
        f.write(str(y))
    plt.scatter(range(len(y)),y)
    plt.show()
    with open('rewards','w') as f:
        f.write(str(r1))
        f.write(str(r2))
    plt.scatter(range(len(r1)),r1,c='g')
    plt.scatter(range(len(r2)), r2, c='r')
    plt.show()
    plt.ylim(-0.3,1)
    plt.scatter(range(len(y)), y)
    plt.show()
    plt.ylim(-5.5,1.5)
    plt.scatter(range(len(r1)), r1, c='g')
    plt.scatter(range(len(r2)), r2, c='r')
    plt.show()


def play():
    random.seed(int(time()))
    p1 = players.QPlayer('Q', [80,80], learning_batch_size=100, gamma=0.9, tau=0.9,
                         batches_to_q_target_switch=100, epsilon=0.0, learning_rate=0.0003)
    p1.restore(os.getcwd() + '/lxo.ckpt')
    p2 = players.Human('N')
    for g in range(2):
        print('STARTING NEW GAME (#{})\n-------------'.format(g))
        if g%2==0:
            game = Game(p1,p2)
            print("Player 1")
        else:
            game = Game(p2,p1)
            print("Player -1")
        while not game.game_status()['game_over']:
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.active_player().name))
            for nth in range(1,9):
                if nth > 1:
                    print('Attempt ',nth)
                state = np.copy(game.board)
                action = int(game.active_player().select_cell(state,nth=nth)) if np.count_nonzero(game.board) > 0 or not isinstance(game.active_player(),players.QPlayer) else random.randint(0,8)
                play_status = game.play(action)
                if not play_status['invalid_move']:
                    break
            if not game.game_status()['game_over']:
                game.next_player()
        print('-------------\nGAME OVER!')
        game.print_board()
        print(game.game_status())
        print('-------------')


train()
