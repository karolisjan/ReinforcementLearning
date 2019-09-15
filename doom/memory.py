'''
    Memory with Priority Experience Replay (PER).
'''
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
        pass

