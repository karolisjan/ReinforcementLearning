'''
    Dueling Double Deep Q-learning Network implementation.

    Input:
        4 frames

    Output:
        Q value for each action
'''
from typing import Sequence, Tuple

import tensorflow as tf
from fire import Fire


class DDDQN(tf.keras.Model):
    '''
        Dueling Double Deep Q-learning Network.
    '''
    def __init__(
        self,
        name: str='DQN',
        **kwargs
    ):
        super(DDDQN, self).__init__(name=name, **kwargs)

        self.conv_layer1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid'
        )

        self.conv_layer1_batchnorm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5
        )

        self.conv_layer2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='valid'
        )

        self.conv_layer2_batchnorm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5
        )

        self.conv_layer3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='valid'
        )

        self.conv_layer3_batchnorm = tf.keras.layers.BatchNormalization(
            epsilon=1e-5
        )

        self.flatten = tf.keras.layers.Flatten()

        self.value_fully_connected = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.elu,
            name='value_fully_connected'
        )

        self.value = tf.keras.layers.Dense(
            units=1,
            activation=None,
            name='value'
        )

        self.advantage_fully_connected = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.elu,
            name='advantage_fully_connected'
        )

        self.advantage = tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.elu,
            name='advantage'
        )

    def call(self, frames, actions):

        x = self.conv_layer1(frames)
        x = self.conv_layer1_batchnorm(x)
        x = self.conv_layer2(x)
        x = self.conv_layer2_batchnorm(x)
        x = self.conv_layer3(x)
        x = self.conv_layer3_batchnorm(x)
        x = self.flatten(x)

        value = self.value_fully_connected(x)
        value = self.value(value)

        advantage = self.advantage_fully_connected(x)
        advantage = self.advantage(advantage)

        output = value + tf.subtract(
            advantage,
            tf.reduce_mean(
                advantage,
                axis=1,
                keepdims=True
            )
        )

        Q = tf.reduce_sum(tf.multiply(output, actions), axis=1)

        return Q


if __name__ == "__main__":
    Fire(DDDQN)

