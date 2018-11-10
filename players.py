from abc import abstractmethod


class Player:
    name = None

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def select_cell(self, board, **kwargs):
        pass

    @abstractmethod
    def receive_reward_and_status(self, reward, game_over, **kwargs):
        pass


class Human(Player):
    def select_cell(self, board, **kwargs):
        cell = input("Select cell to fill:\n678\n345\n012\ncell number: ")
        return cell

    def receive_reward_and_status(self, reward, game_over, **kwargs):
        pass


