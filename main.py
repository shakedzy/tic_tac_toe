import logging
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import players
from game import Game

# Default Q-Player settings
layers_size = [100, 160, 160, 100]
batch_size = 150
batches_to_q_target_switch = 1000
gamma = 0.95
tau = 1
memory_size = 100000
learning_rate = 0.0001


def train(p1_name, p2_name, p1_max_ent, p2_max_ent, p2_novice, num_of_games=1e6, savedir='./models'):
    """
    Initiate a single training process
    :param p1_name: String. Name of player 1 (will be used as file-name)
    :param p2_name: String. Name of player 2 (will be used as file-name)
    :param p1_max_ent: Boolean. Should player 1 use maximum-entropy learning
    :param p2_max_ent: Boolean. Should player 2 use maximum-entropy learning
    :param p2_novice: Boolean. Should player 2 be an instance of players.Novice
    :param num_of_games: Number. Number of games to train on
    :param savedir: String. Path to save trained weights
    """
    random.seed(int(time()*1000))
    tf.reset_default_graph()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Initialize players
    graph1 = tf.Graph()
    graph2 = tf.Graph()

    with graph1.as_default():
        p1 = players.QPlayer(tf.Session(), hidden_layers_size=layers_size, learning_batch_size=batch_size, gamma=gamma,
                             batches_to_q_target_switch=batches_to_q_target_switch, tau=tau, memory_size=memory_size,
                             maximize_entropy=p1_max_ent)
    p1.name = p1_name

    if p2_novice:
        p2 = players.Novice()
    else:
        with graph2.as_default():
            p2 = players.QPlayer(tf.Session(), hidden_layers_size=layers_size, learning_batch_size=batch_size,
                                 gamma=gamma,
                                 batches_to_q_target_switch=batches_to_q_target_switch, tau=tau,
                                 memory_size=memory_size,
                                 maximize_entropy=p2_max_ent)
    p2.name = p2_name

    total_rewards = {p1.name: 0, p2.name: 0}
    costs = {p1.name: [], p2.name: []}  # this will store the costs, so we can plot them later
    rewards = {p1.name: [], p2.name: []}  # same, but for the players total rewards

    # Start playing
    num_of_games = int(num_of_games)
    train_start_time = time()
    for g in range(1,num_of_games+1):
        game = Game(p1,p2) if g%2==0 else Game(p2,p1)  # make sure both players play X and O
        last_phases = {p1.name: None, p2.name: None}  # will be used to store the last state a player was in
        while not game.game_status()['game_over']:
            if isinstance(game.active_player, players.Human):
                game.print_board()
                print("{}'s turn:".format(game.active_player.name))

            # If this is not the first move, store in memory the transition from the last state
            # the active player saw to this one
            state = np.copy(game.board)
            if last_phases[game.active_player.name] is not None:
                memory_element = last_phases[game.active_player.name]
                memory_element['next_state'] = state
                memory_element['game_over'] = False
                game.active_player.add_to_memory(memory_element)

            # Calculate annealed epsilon
            if g <= num_of_games // 4:
                max_eps = 0.6
            elif g <= num_of_games // 2:
                max_eps = 0.1
            else:
                max_eps = 0.05
            min_eps = 0.01
            eps = round(max(max_eps - round(g*(max_eps-min_eps)/num_of_games, 3), min_eps), 3)

            # Play and receive reward
            action = int(game.active_player.select_cell(state, epsilon=eps))
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

            # Store the current state in temporary memory
            last_phases[game.active_player.name] = {'state': state,
                                                      'action': action,
                                                      'reward': r}
            total_rewards[game.active_player.name] += r
            if r == game.winning_reward:
                total_rewards[game.inactive_player.name] += game.losing_reward

            # Activate learning procedure
            cost = game.active_player.learn(learning_rate=learning_rate)
            if cost is not None:
                costs[game.active_player.name].append(cost)

            # Next player's turn, if game hasn't ended
            if not game_over:
                game.next_player()

        # Adding last phase for winning (active) player
        memory_element = last_phases[game.active_player.name]
        memory_element['next_state'] = np.zeros(9)
        memory_element['game_over'] = True
        game.active_player.add_to_memory(memory_element)

        # Adding last phase for losing (inactive) player
        memory_element = last_phases[game.inactive_player.name]
        memory_element['next_state'] = np.zeros(9)
        memory_element['game_over'] = True
        memory_element['reward'] = game.losing_reward if r == game.winning_reward else game.tie_reward
        game.inactive_player.add_to_memory(memory_element)

        # Print statistics
        period = 100.0
        if g % int(period) == 0:
            print('Game: {g} | Number of Trainings: {t1},{t2} | Epsilon: {e} | Average Rewards - {p1}: {r1}, {p2}: {r2}'
                  .format(g=g, p1=p1.name, r1=total_rewards[p1.name]/period,
                          p2=p2.name, r2=total_rewards[p2.name]/period,
                          t1=len(costs[p1.name]), t2=len(costs[p2.name]), e=eps))
            rewards[p1.name].append(total_rewards[p1.name]/period)
            rewards[p2.name].append(total_rewards[p2.name]/period)
            total_rewards = {p1.name: 0, p2.name: 0}

    # Save trained model and shutdown Tensorflow sessions
    training_time = time() - train_start_time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    if seconds < 10:
        seconds = '0{}'.format(seconds)
    print('Training took {m}:{s} minutes'.format(m=minutes, s=seconds))

    # Plot graphs and close sessions
    cost_colors = {p1.name: 'b', p2.name: 'k'}
    reward_colors = {p1.name: 'g', p2.name: 'r'}
    graphs = {p1.name: graph1, p2.name: graph2}
    for pp in [p1,p2]:
        with graphs[pp.name].as_default():
            pp.save('{dir}/{name}.ckpt'.format(dir=savedir, name=pp.name))
            pp.shutdown()

        plt.scatter(range(len(costs[pp.name])), costs[pp.name], c=cost_colors[pp.name])
        plt.title('Cost of player {}'.format(pp.name))
        plt.show()
        plt.scatter(range(len(rewards[pp.name])), rewards[pp.name], c=reward_colors[pp.name])
        plt.title('Average rewards of player {}'.format(pp.name))
        plt.show()

        plt.scatter(range(len(costs[pp.name])), costs[pp.name], c=cost_colors[pp.name])
        plt.title('Cost of player {} [0,1]'.format(pp.name))
        plt.ylim(0,1)
        plt.show()
        plt.scatter(range(len(rewards[pp.name])), rewards[pp.name], c=reward_colors[pp.name])
        plt.title('Average rewards of player {} [-1,1]'.format(pp.name))
        plt.ylim(-1,1)
        plt.show()


def multi_train(n=3):
    """
    Initiate multiple trainings of:
     - DDQN vs DDQN-Maximum-Entropy
     - DDQN vs Novice
     - DDQN-Maximum-Entropy vs Novice
    :param n: Integer. How many multi-trains should be performed
    """
    for i in range(n):
        train(p1_name='Q', p1_max_ent=False,
              p2_name='E', p2_max_ent=True,
              p2_novice=False, savedir='./models/trained_together/{}'.format(i))

        train(p1_name='Q', p1_max_ent=False,
              p2_name='N', p2_max_ent=None,
              p2_novice=True, savedir='./models/trained_against_novice/{}'.format(i))

        train(p1_name='E', p1_max_ent=True,
              p2_name='N', p2_max_ent=None,
              p2_novice=True, savedir='./models/trained_against_novice/{}'.format(i))


def play(model_path, is_max_entropy):
    """
    Play a game against a model
    :param model_path: String. Path to the model
    :param is_max_entropy: Boolean. Does the model uses entropy maximization
    """
    random.seed(int(time()))

    p1 = players.QPlayer(hidden_layers_size=layers_size, learning_batch_size=batch_size, gamma=gamma, tau=tau,
                         batches_to_q_target_switch=batches_to_q_target_switch, memory_size=memory_size,
                         session=tf.Session(), maximize_entropy=is_max_entropy)
    p1.restore(model_path)

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
            if isinstance(game.active_player, players.Human):
                game.print_board()
                print("{}'s turn:".format(game.current_player))
            state = np.copy(game.board)
            # Force Q-Network to select different starting positions if it plays first
            action = int(game.active_player.select_cell(state,epsilon=0.0)) if np.count_nonzero(game.board) > 0 or not isinstance(game.active_player,players.QPlayer) else random.randint(0,8)
            game.play(action)
            if not game.game_status()['game_over']:
                game.next_player()
        print('-------------\nGAME OVER!')
        game.print_board()
        print(game.game_status())
        print('-------------')


def face_off(paths, rng=3, p1_name='Q', p2_name='E'):
    """
    Test different models against each other
    :param paths: List(String). Paths to the models
    :param rng: Integer. How many models in the paths supplied
    :param p1_name: String. Name of player 1
    :param p2_name: String. name of player 2
    :return: Dict. Number of won games per player
    """
    tie = 'TIE'
    results = {p1_name: 0, p2_name: 0, tie: 0}

    for path1 in paths:
        for i in range(rng):
            p1_dir = '{}/{}'.format(path1, i)
            print('Loading player {} [{}]...'.format(p1_name,p1_dir))
            graph1 = tf.Graph()
            with graph1.as_default():
                p1 = players.QPlayer(hidden_layers_size=layers_size, learning_batch_size=batch_size, gamma=gamma, tau=tau,
                                     batches_to_q_target_switch=batches_to_q_target_switch, memory_size=memory_size,
                                     session=tf.Session(), maximize_entropy=False)
                p1.restore('{}/{}.ckpt'.format(p1_dir,p1_name))
                p1.name = p1_name

            for path2 in paths:
                for j in range(rng):
                    p2_dir = '{}/{}'.format(path2, j)
                    print('Loading player {} [{}]...'.format(p2_name,p2_dir))
                    graph2 = tf.Graph()
                    with graph2.as_default():
                        p2 = players.QPlayer(hidden_layers_size=layers_size, learning_batch_size=batch_size, gamma=gamma, tau=tau,
                                             batches_to_q_target_switch=batches_to_q_target_switch, memory_size=memory_size,
                                             session=tf.Session(), maximize_entropy=True)
                        p2.restore('{}/{}.ckpt'.format(p2_dir,p2_name))
                        p2.name = p2_name

                    print('Playing...')
                    print('----------')
                    for g in range(18):
                        if g % 2 == 0:
                            game = Game(p1,p2)
                        else:
                            game = Game(p2,p1)
                        first_cell = g // 2
                        while not game.game_status()['game_over']:
                            state = np.copy(game.board)
                            action = int(game.active_player.select_cell(state,epsilon=0.0)) if np.count_nonzero(game.board) > 0 else first_cell
                            game.play(action)
                            if not game.game_status()['game_over']:
                                game.next_player()
                        winner = game.game_status()['winner']
                        winner_name = game.player1.name if winner == 1 else (game.player2.name if winner == -1 else tie)
                        print('GAME - player X: {p1}, player O: {p2} | First cell: {c} | Winner: {w}'.format(
                            p1=game.player1.name, p2=game.player2.name, c=first_cell, w=winner_name
                        ))
                        results[winner_name] += 1
                    print('----------')

    print('Final results: {}'.format(results))
    s = sum(results.values())
    pct = {k: int(10000*v/s)/100 for k,v in results.items()}
    print('Percents: {}'.format(pct))
    return results


face_off(['./models/trained_against_novice', './models/trained_together'])
