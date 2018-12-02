import logging, os
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import players, dqn


class Game:
    INVALID_REWARD = -10

    board = np.zeros(9)
    current_player = 1
    player1 = players.Player(None)
    player2 = players.Player(None)

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.reset()

    def reset(self):
        self.board = np.zeros(9)
        self.current_player = 1

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
        if self.board[cell] != 0:
            return self.INVALID_REWARD, False
        else:
            self.board[cell] = self.current_player
        status = self.game_status()
        if not status['game_over']:
            self.current_player *= -1
        return status['winner'], status['game_over']

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
    r = []
    random.seed(int(time()*1000))
    tf.reset_default_graph()
    logging.basicConfig(level=logging.WARN, format='%(message)s')
    p1 = players.QPlayer('Q',[80,80],
                         learning_batch_size=100, batches_to_q_target_switch=100,
                         initial_epsilon=0.01, pre_train_steps=5000,
                         memory_size=100000)
    p2 = p1
    game = Game(p1,p2)
    total_reward = 0
    for g in range(1,100001):
        game.reset()
        #print('STARTING NEW GAME (#{})\n-------------'.format(g))
        while not game.game_status()['game_over']:
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.active_player().name))
            state = game.current_player * game.board
            action = int(game.active_player().select_cell(state)) if np.count_nonzero(game.board) > 0 else random.randint(0,8)
            reward, game_over = game.play(action)
            if reward != game.INVALID_REWARD:
                reward *= game.current_player
            if game.current_player == 1:
                total_reward += reward
            next_state = game.current_player * game.board if not game_over else np.zeros(9)
            game.active_player().memory.append({'state': state, 'action': action,
                                                'reward': reward, 'next_state': next_state,
                                                'game_over': game_over})
            if reward != game.INVALID_REWARD:
                game.inactive_player().memory.append({'state': -state, 'action': action,
                                                      'reward': -reward, 'next_state': -next_state,
                                                      'game_over': game_over})
            cost = game.active_player().learn()
            if cost is not None:
                y.append(cost)
        r.append(total_reward)
        if g % 100 == 0:
            print('Game: {g} | Reward: {r}'.format(g=g,r=total_reward))
        total_reward = 0
        #print('-------------\nGAME OVER!')
        #game.print_board()
        #print(game.game_status())
        #print('-------------')
    p1.save(os.getcwd() + '/q.ckpt')
    for pp in [p1,p2]:
        pp.shutdown()
    with open('costs','w') as f:
        f.write(str(y))
    plt.scatter(range(len(y)),y)
    plt.show()
    with open('rewards','w') as f:
        f.write(str(r))
    plt.scatter(range(len(r)),r,c='g')
    plt.show()


def play():
    random.seed(int(time()))
    p1 = players.QPlayer('Q',[80,100,80],learning_batch_size=100,
                         batches_to_q_target_switch=100,initial_epsilon=0.0,
                         pre_train_steps=0)
    p1.restore(os.getcwd() + '/q.ckpt')
    p2 = players.Human('H')
    game = Game(p1,p2)
    for g in range(5):
        game.reset()
        total_reward = 0
        print('STARTING NEW GAME (#{})\n-------------'.format(g))
        while not game.game_status()['game_over']:
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.active_player().name))
            state = game.current_player * game.board
            action = int(game.active_player().select_cell(state)) if np.count_nonzero(game.board) > 0 or not isinstance(game.active_player(),players.QPlayer) else random.randint(0,8)
            reward, game_over = game.play(action)
            total_reward += reward
        print('-------------\nGAME OVER!')
        game.print_board()
        print(game.game_status())
        print('Agent\'s reward: ', total_reward)
        print('-------------')


train()

