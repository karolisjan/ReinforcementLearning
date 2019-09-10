from typing import Sequence, Tuple

import tensorflow as tf
from fire import Fire


class DQN(tf.keras.Model):
    def __init__(
        self,
        name: str='DQN',
        **kwargs
    ):
        super(DQN, self).__init__(name=name, **kwargs)

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

        self.fully_connected = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.elu
        )

        self.output = tf.keras.layers.Dense(units=3, activation=None)

    def call(self, inputs):
        x = self.conv_layer1(inputs)
        x = self.conv_layer1_batchnorm(x)
        x = self.conv_layer2(x)
        x = self.conv_layer2_batchnorm(x)
        x = self.conv_layer3(x)
        x = self.conv_layer3_batchnorm(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return self.output(x)


if __name__ == "__main__":
    Fire(DQN)

