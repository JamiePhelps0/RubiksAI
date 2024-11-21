import time
import numpy as np
import tensorflow as tf
import os

window = 6

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


cmd = "/home/jamie/IdeaProjects/RubiksNew/AlphaZero/EnvC"

inp = tf.keras.layers.Input((20, 24))
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(12)(x)

net = tf.keras.models.Model(inp, x)

net.load_weights('/home/jamie/IdeaProjects/RubiksNew/DQN/FC_q.h5')
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
batch_size = 128


def get_data(idx):
    ds_states = []
    ds_labels = []
    for j in range(max(0, idx + 1 - window), idx + 1):
        with open(f"/home/jamie/IdeaProjects/RubiksNew/AlphaZero/DataFiles/Dataset{j}/data0.txt", "r+") as file:
            out = file.read()
        out = out.split('\n')[:-1]
        labels, states = [i.split(':')[0] for i in out], [i.split(':')[1] for i in out]
        labels = [i.split(';')[:-1] for i in labels]
        states = [i.split(';')[:-1] for i in states]
        labels_all = np.array([[float(i) for i in j] for j in labels])
        states_all = np.array([[int(i) for i in j] for j in states])
        for i in range(1, 16):
            with open(f"/home/jamie/IdeaProjects/RubiksNew/AlphaZero/DataFiles/Dataset{j}/data{i}.txt", "r+") as file:
                out = file.read()
            out = out.split('\n')[:-1]
            labels, states = [i.split(':')[0] for i in out], [i.split(':')[1] for i in out]
            labels = [i.split(';')[:-1] for i in labels]
            states = [i.split(';')[:-1] for i in states]
            labels = np.array([[float(i) for i in j] for j in labels])
            states = np.array([[int(i) for i in j] for j in states])
            labels_all = np.vstack([labels, labels_all])
            states_all = np.vstack([states, states_all])
        ds_states.append(states_all)
        ds_labels.append(labels_all)
    return np.vstack(ds_states), np.vstack(ds_labels)


def batch_generator(idx):
    states, labels = get_data(idx)
    indexes = np.arange(len(states))
    print(len(states))
    np.random.shuffle(indexes)
    batch = []
    for idx in indexes:
        batch.append([states[idx], labels[idx]])
        if len(batch) == batch_size:
            yield batch
            batch = []


for i in range(10000):
    print(i)
    with open(f"/AlphaZero/DataFiles/idx.txt", "w+") as file:
        file.write(str(i))
    while os.path.isfile("/home/jamie/IdeaProjects/RubiksNew/AlphaZero/l.lock"):
        try:
            os.remove("/home/jamie/IdeaProjects/RubiksNew/AlphaZero/l.lock")
        except PermissionError:
            pass
    while not os.path.isfile("/home/jamie/IdeaProjects/RubiksNew/AlphaZero/l.lock"):
        time.sleep(0.1)
    for _ in range(4):
        losses = []
        for batch in batch_generator(i):
            batch = np.array(batch)
            states = np.array(batch[:, 0].tolist())
            targets = np.array(batch[:, 1].tolist())

            states = tf.one_hot(states, depth=24)

            with tf.GradientTape() as tape:
                out = net(states)
                loss = tf.reduce_mean(tf.square(targets - out))

            grads = tape.gradient(target=loss, sources=net.trainable_variables)
            opt.apply_gradients(zip(grads, net.trainable_variables))
            losses.append(loss)
        print(np.mean(losses))
        losses = []
    tf.saved_model.save(net, "/DQN/savedmodel")
    time.sleep(0.5)


