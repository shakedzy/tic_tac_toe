import numpy as np


class Game:
    """
    Tic-Tac-Toe game class
    """
    board = np.zeros(9)
    current_player = 1  # first player is 1, second player is -1
    player1 = None
    player2 = None

    _invalid_move_played = False

    def __init__(self, player1, player2,
                 winning_reward=1,
                 losing_reward=-1,
                 tie_reward=0,
                 invalid_move_reward=-10):
        self.player1 = player1
        self.player2 = player2
        self.player1.player_id = 1
        self.player2.player_id = -1
        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.invalid_move_reward = invalid_move_reward
        self.tie_reward = tie_reward
        self.reset()

    @property
    def active_player(self):
        if self.current_player == 1:
            return self.player1
        else:
            return self.player2

    @property
    def inactive_player(self):
        if self.current_player == -1:
            return self.player1
        else:
            return self.player2

    def reset(self):
        self.board = np.zeros(9)
        self.current_player = 1
        self._invalid_move_played = False

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
