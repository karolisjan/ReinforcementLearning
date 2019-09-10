from typing import Sequence, Tuple

import tensorflow as tf
from fire import Fire


class DQN(tf.keras.Model):
    def __init__(
        self,
        state_size: Tuple[int],
        action_size: int,
        learning_rate: float,
        name: str='DQN',
        **kwargs
    ):
        super(DQN, self).__init__(name=name, **kwargs)

        self.inputs = tf.placeholder(
            dtype=tf.float32,
            shape=[None, *state_size],
            name='inputs'
        )


if __name__ == "__main__":
    Fire(DQN)

