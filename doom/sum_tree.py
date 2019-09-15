'''
    Priority Experience Replay implementation.
'''
import numpy as np


class SumTree:
    '''
        Efficient implementation of Priority Experience
        Replay using an unsorted Sum Tree.

        Modified from https://tinyurl.com/yyqvqupm
    '''
    def __init__(self, capacity: int):
        self.__capacity = capacity
        self.__tree = np.zeros(2 * capacity - 1)
        self.__data = np.zeros(capacity, dtype=object)
        self.__data_pointer = 0

    @property
    def capacity(self):
        return self.__capacity

    @property
    def tree(self):
        return self.__tree

    @property
    def data(self):
        return self.__data

    def add(self, priority: float, experience: object):
        '''
            Adds priority score in the sum tree and
            the experience in the data.
        '''
        index = self.__data_pointer + self.capacity - 1
        self.data[self.__data_pointer] = experience
        self.update(index, priority)
        self.__data_pointer += 1

        if self.__data_pointer >= self.capacity:
            self.__data_pointer = 0

    def update(self, index: int, priority: float):
        delta = priority - self.tree[index]
        self.tree[index] = priority

        # Propogate the change in priority through the tree
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get_leaf(self, v):
        index = None
        parent = 0

        while True:
            left_child = parent * 2 + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                index = parent
                break

            # Searching for a higher priority node
            if v <= self.tree[left_child]:
                parent = left_child
            else:
                v -= self.tree[left_child]
                parent = right_child

        data_index = index - self.capacity + 1
        return index, self.tree[index], self.data[data_index]

    @property
    def priority(self):
        return self.tree[0]
