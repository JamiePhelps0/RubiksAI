import multiprocessing
import Env
import PER
from Env import RubiksEnv
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import pickle

tf.random.set_seed(2022)
np.random.seed(2022)

batch_size = 20000
discount = 0.99
q_lr = 0.00005


class DuelingNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(2048)
        self.fc2 = tf.keras.layers.Dense(2048)
        self.fc3 = tf.keras.layers.Dense(2048)
        self.fc4 = tf.keras.layers.Dense(2048)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.adv_fc = tf.keras.layers.Dense(512)
        self.adv = tf.keras.layers.Dense(12)

        self.v_fc = tf.keras.layers.Dense(512)
        self.v = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = tf.keras.activations.swish(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = tf.keras.activations.swish(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = tf.keras.activations.swish(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = tf.keras.activations.swish(x)
        x = self.bn4(x)

        adv = self.adv_fc(x)
        adv = self.adv(adv)

        v = self.v_fc(x)
        v = self.v(v)

        out = v + (adv - tf.reduce_mean(adv))
        return out


def get_net():
    inp = tf.keras.layers.Input((20, 24))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(12)(x)

    model = tf.keras.models.Model(inp, x)

    # def res_connection(x):
    #     res = x
    #     x = tf.keras.layers.Conv2D(channels[i], 3, padding='same', activation=tf.keras.activations.swish)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Conv2D(channels[i], 3, padding='same', activation=tf.keras.activations.swish)(x)
    #     x = tf.keras.layers.Add()([res, x])
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     return x
    #
    # inp = tf.keras.layers.Input((20, 24))
    # x = tf.keras.layers.Reshape((20, 24, 1))(inp)
    # x = tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.keras.activations.swish)(x)
    # channels = [64, 256, 512]
    # for i in range(len(channels) - 1):
    #     x = res_connection(x)
    #     x = res_connection(x)
    #     x = tf.keras.layers.Conv2D(channels[i + 1], 3, 2, padding='same', activation=tf.keras.activations.swish)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(12)(x)
    #
    # model = tf.keras.models.Model(inp, x)

    print(model.summary())
    return model


def get_file(idx, q=None):
    with open(f'/home/jamie/IdeaProjects/RubiksNew/Datasets/3x3x3_State_DS/data{idx}.pkl', 'rb') as file:
        data = pickle.load(file)
    data = np.array(data)
    data = np.array(np.split(data, len(data) // 30))
    if q:
        q.put(data)
    else:
        return data


def batch_generator(n_step=1, num_files=6):
    data = get_file(0)
    q = multiprocessing.Queue()
    files = np.arange(1, num_files)
    np.random.shuffle(files)
    batch = []
    for file in files:
        next_file = multiprocessing.Process(target=get_file, args=(file, q))
        next_file.start()
        game_idx = np.arange(len(data))
        np.random.shuffle(game_idx)
        for game in game_idx:
            for k in range(len(data[game])):
                if k <= n_step:
                    batch.append([data[game][k][0], data[game][k][1], -(k + 1), data[game][0][2], True])
                else:
                    batch.append([data[game][k][0], data[game][k][1], -n_step, data[game][k - n_step][2], False])
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        data = q.get()
        next_file.join()
    game_idx = np.arange(len(data))
    np.random.shuffle(game_idx)
    for game in game_idx:
        for k in range(len(data[game])):
            if k <= n_step:
                batch.append([data[game][k][0], data[game][k][1], -(k + 1), data[game][0][2], True])
            else:
                batch.append([data[game][k][0], data[game][k][1], -n_step, data[game][k - n_step][2], False])
            if len(batch) == batch_size:
                yield batch
                batch = []


def test(policy, scram, tests=100):
    env = RubiksEnv()
    solved = []
    for i in range(tests):
        env.reset(scram)
        total = 0
        done = 0
        while total < 2 * scram:
            state = np.reshape(env.state(), (1, 3 ** 3))
            idxs = np.where(state[0] == -1)[0]
            state = np.delete(state, idxs, axis=-1)
            state = tf.one_hot(state, depth=24)
            # state = tf.one_hot(np.array([env.state()]), depth=24)
            action = np.argmax(policy.q_net(state))
            env.take_action(action)
            if env.is_solved():
                done = 1
                break
            total += 1
        solved.append(done)
    print(f'test{scram}: {np.mean(solved)}')


def solve(env, net):
    total = 0
    while total < 20:
        action = np.argmax(net(np.array([tf.one_hot(env.state(), depth=6)])))
        env.take_action(action)
        if env.is_solved():
            return env
        total += 1
    return False


class DQN_Agent:
    def __init__(self):
        self.q_net = get_net()
        self.target_net = get_net()
        for t, e in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            t.assign(e)
        # self.q_net(np.zeros((1, 20, 24)))
        # self.target_net(np.zeros((1, 20, 24)))
        # self.load_weights()
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=q_lr)

    def act(self):
        pass

    def save_weights(self):
        self.q_net.save_weights('Conv_q.h5')
        self.target_net.save_weights('Conv_t.h5')

    def load_weights(self):
        self.q_net.load_weights('Conv_q.h5')
        self.target_net.load_weights('Conv_t.h5')

    def search(self, state, top_k=3, depth=3):
        envs = [state]
        for _ in range(depth):
            new_envs = []
            for i in envs:
                actions = np.argsort(self.q_net(np.array([tf.one_hot(i.state(), depth=20)])), axis=-1)[:, -top_k:][0]
                for j in actions:
                    new = i.copy()
                    new.take_action(j)
                    new_envs.append(new)
            envs = new_envs

        best = [-100, None]
        for env in envs:
            q_max = np.max(self.q_net((np.array([tf.one_hot(env.state(), depth=20)]))))
            if q_max > best[0]:
                best = [q_max, env]
        return best

    def fill_buffer(self, num_games, scram_num=25, buffer=None):
        env = Env.RubiksEnv()
        for _ in range(num_games):
            env.reset(scram_num)
            # state = np.reshape(env.state(), (1, 3 ** 3))
            # idxs = np.where(state[0] == -1)[0]
            # state = np.delete(state, idxs, axis=-1)
            state = env.state()
            k = 0
            while (not env.is_solved()) and (k < int(1.5 * scram_num)):
                if np.random.uniform() < 0.02:
                    action = random.randint(0, 11)
                else:
                    action = np.argmax(self.q_net(tf.one_hot(np.array([state]), depth=24)))
                # env.take_action(action)
                # next_state = np.reshape(env.state(), (1, 3 ** 3))
                # idxs = np.where(next_state[0] == -1)[0]
                # next_state = np.delete(next_state, idxs, axis=-1)
                next_state = env.state()
                buffer.store(state, action, -1, next_state, env.is_solved())
                state = next_state
                k += 1

    def learn_ds(self, iterations, num_files=6, limit=9e9):
        losses = []
        for _ in range(iterations):
            loss = 0.0
            iterator = tqdm(enumerate(batch_generator(n_step=3, num_files=num_files)), total=min(15e6 * num_files // batch_size, limit), desc='loss: 0.0\t')
            for j, batch in iterator:
                iterator.set_description(f'loss: {round(float(loss), 5)}\t')
                batch = np.array(batch)
                states = np.array(batch[:, 0].tolist())
                actions = np.array(batch[:, 1].tolist())
                rewards = np.array(batch[:, 2].tolist())
                next_states = np.array(batch[:, 3].tolist())
                dones = np.array(batch[:, 4].tolist())

                states = np.reshape(states, (batch_size, 3 ** 3))
                idxs = np.where(states[0] == -1)[0]
                states = np.delete(states, idxs, axis=-1)
                states = tf.one_hot(states, depth=24)

                next_states = np.reshape(next_states, (batch_size, 3 ** 3))
                idxs = np.where(next_states[0] == -1)[0]
                next_states = np.delete(next_states, idxs, axis=-1)
                next_states = tf.one_hot(next_states, depth=24)

                q_next = self.q_net(next_states)
                next_a = np.argmax(q_next, axis=-1)
                q_target_next = self.target_net(next_states)
                # print(q_target_next.shape, next_a.shape, next_states.shape, states.shape, dones.shape)
                y = rewards + discount * (1 - dones) * tf.reduce_sum(tf.one_hot(next_a, depth=12) * q_target_next, axis=-1)
                with tf.GradientTape() as tape:
                    q = tf.reduce_sum(self.q_net(states) * tf.one_hot(actions, depth=12), axis=-1)
                    td = y - q
                    loss = tf.reduce_mean(tf.square(td))
                grads = tape.gradient(target=loss, sources=self.q_net.trainable_variables)
                self.optimiser.apply_gradients(zip(grads, self.q_net.trainable_variables))
                losses.append(loss)
                for t, e in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
                    t.assign(t * (1 - 0.01) + e * 0.01)
                if j % 1000 == 0:
                    self.save_weights()
                    print(np.mean(losses))
                    losses = []
                    test(self, 10, tests=50)
                if j == limit:
                    break
            self.save_weights()
            [test(agent, i, tests=10) for i in range(1, 21)]

        return np.mean(losses)

    def learn_buff(self, num_batches, buffer):
        losses = []
        for i in range(num_batches):
            batch, w, idxs = buffer.sample(batch_size=batch_size)
            states, actions, rewards, next_states, dones = batch
            states = tf.one_hot(states, depth=24)
            next_states = tf.one_hot(next_states, depth=24)

            q_next = self.q_net(next_states)
            next_a = np.argmax(q_next, axis=-1)
            q_target_next = self.target_net(next_states)
            # print(q_target_next.shape, next_a.shape, next_states.shape, states.shape, dones.shape)
            y = rewards + discount * (1 - dones) * tf.reduce_sum(tf.one_hot(next_a, depth=12) * q_target_next, axis=-1)
            with tf.GradientTape() as tape:
                q = tf.reduce_sum(self.q_net(states) * tf.one_hot(actions, depth=12), axis=-1)
                td = y - q
                loss = tf.reduce_mean(tf.square(td * w))
            buffer.update_priorities(idxs, np.abs(td))
            grads = tape.gradient(target=loss, sources=self.q_net.trainable_variables)
            self.optimiser.apply_gradients(zip(grads, self.q_net.trainable_variables))
            losses.append(loss)
            for t, e in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
                t.assign(t * (1 - 0.001) + e * 0.001)
        return np.mean(losses)

    def online_learn(self, steps, buffer):
        for i in range(steps):
            self.fill_buffer(50, 15, buffer)
            self.fill_buffer(50, 40, buffer)
            print(self.learn_buff(60, buffer))
            if i % 10 == 0:
                self.save_weights()
                test(self, 10, tests=250)


if __name__ == '__main__':
    # buffer = Buffer()
    agent = DQN_Agent()

    # tf.saved_model.save(agent.q_net, "savedmodel/")
    # agent.q_net = tf.saved_model.load("/home/jamie/IdeaProjects/RubiksNew/DQN/savedmodel/")
    test(agent, 5, 100)
    # exit()

    # agent.fill_buffer(500, scram_num=30)
    # agent.online_learn(10000000)
    # test(agent, 10, tests=1000)
    # agent.learn_ds(10)
    agent.learn_ds(5)
    [test(agent, i, tests=1000) for i in range(1, 21)]
    exit()

    # test(agent, 10)

    buffer = PER.PrioritizedReplayBuffer(10_000_000)
    # i = 0
    # for i, batch in enumerate(batch_generator(n_step=3)):
    #     if i % 100 == 0:
    #         print(i)
    #     if i == 100:
    #         break
    #     batch = np.array(batch)
    #     states = np.array(batch[:, 0].tolist())
    #     actions = np.array(batch[:, 1].tolist())
    #     rewards = np.array(batch[:, 2].tolist())
    #     next_states = np.array(batch[:, 3].tolist())
    #     dones = np.array(batch[:, 4].tolist())
    #     for s, a, r, sp, d in zip(states, actions, rewards, next_states, dones):
    #         buffer.store(s, a, r, sp, d)
    # print('start')

    agent.fill_buffer(100, 10, buffer)
    agent.fill_buffer(100, 50, buffer)
    print(buffer.real_size)

    agent.online_learn(100000, buffer)

    for i in range(15, 18):
        env = Env.RubiksEnv()
        total = []
        for _ in range(100):
            env.reset(i)
            num, solution = agent.search(env, top_k=4, depth=2)
            total.append(1 if solve(solution, agent.q_net) is not False else 0)

        print(f'test{i}: {np.mean(total)}')


    test(agent, 20, tests=1000)
    #
    # print(total)





    # agent.save_weights()
    # agent.online_learn(1000)
    # [test(agent, i) for i in range(1, 11)]

    # test(agent, 10)
    # test(agent, 20)



