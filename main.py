import logging
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import players
from game import Game


def train():
    y = []
    r1 = []
    r2 = []
    random.seed(int(time()*1000))
    tf.reset_default_graph()
    logging.basicConfig(level=logging.WARN, format='%(message)s')
    p1 = players.QPlayer([100,160,160,100],
                         learning_batch_size=150, batches_to_q_target_switch=1000,
                         gamma=0.95, tau=0.95, memory_size=100000)
    p1.name = 'Q'
    p2 = players.Novice()
    p2.name = 'N'
    total_rewards = {p1.name: 0, p2.name: 0}
    num_of_games = 400000
    for g in range(1,num_of_games+1):
        game = Game(p1,p2) if g%2==0 else Game(p2,p1)
        last_phases = {p1.name: None, p2.name: None}
        while not game.game_status()['game_over']:
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.active_player().name))

            state = np.copy(game.board)
            if last_phases[game.active_player().name] is not None:
                memory_element = last_phases[game.active_player().name]
                memory_element['next_state'] = state
                memory_element['game_over'] = False
                game.active_player().add_to_memory(memory_element)

            if g <= num_of_games // 4:
                max_eps = 0.6
            elif g <= num_of_games // 2:
                max_eps = 0.01
            else:
                max_eps = 0.001
            min_eps = 0.01 if g <= num_of_games // 2 else 0.0
            eps = round(max(max_eps - round(g*(max_eps-min_eps)/num_of_games, 3), min_eps), 3)

            action = int(game.active_player().select_cell(state,epsilon=eps))
            play_status = game.play(action)
            game_over = play_status['game_over']
            if play_status['invalid_move']:
                r = game.invalid_move_reward
            elif game_over:
                if play_status['winner'] == 0:
                    r = game.tie_reward
                else:
                    r = game.winning_reward
            else:
                r = 0
            last_phases[game.active_player().name] = {'state': state,
                                                      'action': action,
                                                      'reward': r}
            total_rewards[game.active_player().name] += r
            cost = game.active_player().learn(learning_rate=0.0001)
            if cost is not None:
                y.append(cost)
            if not game_over:
                game.next_player()

        # adding last phase for winning (active) player
        memory_element = last_phases[game.active_player().name]
        memory_element['next_state'] = np.zeros(9)
        memory_element['game_over'] = True
        game.active_player().add_to_memory(memory_element)

        # adding last phase for losing (inactive) player
        memory_element = last_phases[game.inactive_player().name]
        memory_element['next_state'] = np.zeros(9)
        memory_element['game_over'] = True
        memory_element['reward'] = game.losing_reward
        game.inactive_player().add_to_memory(memory_element)

        if g % 100 == 0:
            print('Game: {g} | Number of Trainings: {t} | Epsilon: {e} | Average Rewards - {p1}: {r1}, {p2}: {r2}'
                  .format(g=g, p1=p1.name, r1=total_rewards[p1.name]/100.0,
                          p2=p2.name, r2=total_rewards[p2.name]/100.0,
                          t=len(y), e=eps))
            r1.append(total_rewards[p1.name]/100.0)
            r2.append(total_rewards[p2.name]/100.0)
            total_rewards = {p1.name: 0, p2.name: 0}
    p1.save('./models/q3.ckpt')
    for pp in [p1,p2]:
        pp.shutdown()
    plt.scatter(range(len(y)),y)
    plt.show()
    plt.scatter(range(len(r1)),r1,c='g')
    plt.show()
    plt.scatter(range(len(r2)), r2, c='r')
    plt.show()


def play():
    random.seed(int(time()))
    p1 = players.QPlayer([100,160,160,100], learning_batch_size=100, gamma=0.9, tau=0.9,
                         batches_to_q_target_switch=100)
    p1.restore('./models/q.ckpt')
    p2 = players.Human()
    for g in range(4):
        print('STARTING NEW GAME (#{})\n-------------'.format(g))
        if g%2==0:
            game = Game(p1,p2)
            print("Computer is X (1)")
        else:
            game = Game(p2,p1)
            print("Computer is O (-1)")
        while not game.game_status()['game_over']:
            if isinstance(game.active_player(), players.Human):
                game.print_board()
                print("{}'s turn:".format(game.current_player))
            for nth in range(1,9):
                if nth > 1:
                    print('Attempt ',nth)
                state = np.copy(game.board)
                action = int(game.active_player().select_cell(state,epsilon=0.0,nth=nth)) if np.count_nonzero(game.board) > 0 or not isinstance(game.active_player(),players.QPlayer) else random.randint(0,8)
                play_status = game.play(action)
                if not play_status['invalid_move']:
                    break
            if not game.game_status()['game_over']:
                game.next_player()
        print('-------------\nGAME OVER!')
        game.print_board()
        print(game.game_status())
        print('-------------')


play()
