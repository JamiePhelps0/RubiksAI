import numpy as np
import tensorflow as tf
import os
from timeit import default_timer

inp = tf.keras.layers.Input((20, 24))
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(12)(x)

net = tf.keras.models.Model(inp, x)

net.load_weights('C:/Users/Jamie Phelps/Documents/RubiksNew/DQN/FC_q.h5')


while True:
    if os.path.isfile("C:/Users/Jamie Phelps/Documents/RubiksNew/AlphaZero/l.lock"):
        with open("C:/Users/Jamie Phelps/Documents/RubiksNew/AlphaZero/data.txt", 'r+') as file:
            data = file.read()
        arr = np.array([int(i) for i in data.split(';')[:-1]])

        out = np.array(net(np.array([tf.one_hot(arr, depth=24)])))[0]
        with open("C:/Users/Jamie Phelps/Documents/RubiksNew/AlphaZero/data.txt", 'w+') as file:
            file.write("".join(["".join([str(i), '\n']) for i in out]))

        while os.path.isfile("C:/Users/Jamie Phelps/Documents/RubiksNew/AlphaZero/l.lock"):
            try:
                os.remove("C:/Users/Jamie Phelps/Documents/RubiksNew/AlphaZero/l.lock")
            except PermissionError:
                pass









