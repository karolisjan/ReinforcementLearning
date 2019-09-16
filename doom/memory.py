'''
    Memory with Priority Experience Replay (PER).
'''
from typing import Sequence

import numpy as np

from sum_tree import SumTree


class Memory:
    def __init__(
        self,
        capacity: int,
        per_e: float=0.01,
        per_a: float=0.6,
        per_b: float=0.4,
        per_b_increment: float=0.001
    ):
        '''
            SumTree-based PER Memory.

            Arguments
            ---------
                capacity, int: SumTree capacity.

                per_e, float: min probability.

                per_a, float: trade-off taking high priority
                              experience and random sampling.

                per_b, float: importance-sampling.

                per_b_increment, float: value used to increment
                                        per_b.
        '''
        self.__sum_tree = SumTree(capacity)
        self.__per_e = per_e
        self.__per_a = per_a
        self.__per_b = per_b
        self.__per_b_increment = per_b_increment
        self.__error_ceil = 1.0

    def add(self, experience: object):
        '''
            Adds a new experience to the sum tree.
        '''
        priority = np.max(self.__sum_tree.tree[-self.__sum_tree.capacity:])

        if np.isclose(priority, 0, atol=0):
            priority = self.__error_ceil

        self.__sum_tree.add(priority, experience)

    def sample(self, n: int):
        replay = []

        indices = np.empty((n, ), dtype=int)
        is_weights = np.empty((n, 1), dtype=float)
        segment = self.__sum_tree.total_priority / n
        self.__per_b = np.min([1.0, self.__per_b + self.__per_b_increment])

        p_min = np.min(
            self.__sum_tree.tree[-self.__sum_tree.capacity:],
            self.__sum_tree.total_priority
        )

        max_weight = np.power(p_min * n, -self.__per_b)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority, experience = self.__sum_tree.tree.get_leaf(v)
            p_j = priority / self.__sum_tree.tree.total_priority
            is_weights[i, 0] = np.power(p_j * n, -self.__per_b) / max_weight
            indices[i] = index
            replay.append([experience])

        return indices, replay, is_weights

    def update(self, indices: Sequence[int], error: float):
        error += self.__per_e
        clipped_error = np.min([error, self.__error_ceil])
        ps = np.power(clipped_error, self.__per_a)

        for index, p in zip(indices, ps):
            self.__sum_tree.tree.update(index, p)
