import random
from collections import deque
from abc import abstractmethod


class MemoryTemplate:
    """
    Memory abstract class
    """
    _counter = 0

    def __init__(self, seed):
        if seed is not None:
            random.seed(seed)

    @property
    def counter(self):
        return self._counter

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def append(self, element):
        # remember to call to _inc_counter when appending
        pass

    @abstractmethod
    def sample(self, n, or_less):
        pass

    def _inc_counter(self, inc_by=1):
        self._counter += inc_by

    def _get_n_or_less(self, n, or_less):
        if or_less and n > self._counter:
            result = self._counter
        else:
            result = n
        return result


class ExperienceReplayMemory(MemoryTemplate):
    """
    A cyclic-buffer Experience Replay memory
    """
    _memory = None

    def __init__(self, size, seed=None):
        """
        Create a new Experience Replay Memory
        :param size: memory size
        :param seed: random seed to be used (will override random.seed)
        """
        super(ExperienceReplayMemory, self).__init__(seed)
        self._memory = deque(maxlen=size)

    def __len__(self):
        return len(self._memory)

    def append(self, element):
        self._memory.append(element)
        self._inc_counter()

    def sample(self, n, or_less=False):
        n = self._get_n_or_less(n, or_less)
        return random.sample(self._memory, n)


class ReservoirSamplingMemory(MemoryTemplate):
    """
    Reservoir Sampling based memory buffer
    """
    _memory = list()
    _max_size = 0

    def __init__(self, size, seed=None):
        """
        Create a new Reservoir Sampling Memory
        :param size: memory size
        :param seed: random seed to be used (will override random.seed)
        """
        super(ReservoirSamplingMemory, self).__init__(seed)
        self._max_size = size

    def __len__(self):
        return len(self._memory)

    def append(self, element):
        if len(self._memory) < self._max_size:
            self._memory.append(element)
        else:
            i = int(random.random() * self._counter)
            if i < self._max_size:
                self._memory[i] = element

    def sample(self, n ,or_less=False):
        n = self._get_n_or_less(n,or_less)
        return random.sample(self._memory, n)
