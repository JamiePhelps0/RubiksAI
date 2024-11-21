import pickle
from Env import RubiksEnv
from multiprocessing import Pool
from timeit import default_timer
import tensorflow as tf
import numpy as np

num_actions = 12
c = 2.5


class Node:
    def __init__(self, parent, game: RubiksEnv, action: int, values: np.ndarray):
        self.game = game
        self.parent = parent
        self.action = action
        self.solved_child = -1
        if game.is_solved():
            self.parent.solved_child = action
            self.q_values = [-1]
        else:
            self.UCB = None
            self.action_counts = [0 for _ in range(num_actions)]
            self.children = [None for _ in range(num_actions)]
            self.q_values = values

    def select(self):
        self.UCB = self.q_values + c * (np.sqrt(sum(self.action_counts)) / (1 + np.array(self.action_counts)))
        if self.solved_child != -1:
            self.UCB[self.solved_child] = -9999
        return np.argmax(self.UCB)

    def get_new_game(self, action):
        new_game = self.game.copy()
        new_game.take_action(action)
        return new_game

    def create_child(self, game, action, values):
        self.children[action] = Node(self, game, action, values)


def batch_search(games: list, simulations=800):
    def get_values(game_list):
        states = np.array([env.state() for env in game_list])
        states = np.reshape(states, (len(game_list), 3 ** 3))
        idxs = np.where(states[0] == -1)[0]
        states = np.delete(states, idxs, axis=-1)
        states = tf.one_hot(states, depth=24)
        return np.array(net(states))
    values = get_values(games)
    start_nodes = [Node(None, game, 0, value) for game, value in zip(games, values)]
    for _ in range(simulations):
        actions = [start.select() for start in start_nodes]
        current = [start for start in start_nodes]
        leafs = []
        new_a = []
        for curr, a in zip(current, actions):
            while curr.children[a] is not None:
                curr.action_counts[a] += 1
                curr = curr.children[a]
                a = curr.select()
            leafs.append(curr)
            new_a.append(a)
        new_states = [curr.get_new_game(a) for curr, a in zip(leafs, new_a)]
        values = get_values(new_states)
        [curr.create_child(new_state, a, value) for curr, new_state, a, value in zip(leafs, new_states, new_a, values)]
        current = [curr.children[a] for curr, a in zip(leafs, new_a)]
        for curr in current:
            while curr.parent is not None:
                curr.parent.q_values[curr.action] = -1 + np.max(curr.q_values)
                curr = curr.parent
    return start_nodes


def search(game, simulations=800, training=True):
    def get_q_values(game):
        state = np.reshape(game.state(), (1, 3 ** 3))
        idxs = np.where(state[0] == -1)[0]
        state = np.delete(state, idxs, axis=-1)
        state = tf.one_hot(state, depth=24)
        return np.array(net(state))[0]
    game.actions_taken = []
    values = get_q_values(game)
    start = Node(None, game, 0, values)
    for _ in range(simulations):
        a = start.select()
        current = start
        while current.action_counts[a] > 0:
            current.action_counts[a] += 1
            current = current.children[a]
            a = current.select()
        new_state = current.get_new_game(a)
        current.create_child(new_state, a, get_q_values(new_state))
        current.action_counts[a] += 1
        current = current.children[a]
        # if current.solved and not training:
        #     actions = []
        #     while current.parent is not None:
        #         actions.append(current.action)
        #         current = current.parent
        #     return start, reversed(actions)
        while current.parent is not None:
            current.parent.q_values[current.action] = -1 + np.max(current.q_values)
            current = current.parent
    return start, None


def get_dataset(ds_size, simulations=800, parallel=256, depth=20):
    dataset = []
    environments = [RubiksEnv() for _ in range(parallel)]
    [env.reset(depth) for env in environments]
    step_counts = [0 for _ in range(parallel)]
    while len(dataset) < ds_size:
        reset_envs = [0 if (i < depth) and not env.is_solved() else 1 for i, env in zip(step_counts, environments)]
        [env.reset(depth) if i else None for env, i in zip(environments, reset_envs)]
        nodes = batch_search(environments.copy(), simulations=simulations)
        targets = [node.q_values for node in nodes]
        for target, environment in zip(targets, environments):
            dataset.append([environment.state(), target])
        [env.take_action(np.argmax(q_value)) for env, q_value in zip(environments, targets)]
        step_counts = [i + 1 for i in step_counts]
    with open('C:/Users/Jamie Phelps/Documents/RubiksNew/Datasets/MCTS/ds.pkl', 'wb') as file:
        pickle.dump(dataset, file)


def batch_generator(batch_size=128):
    with open('C:/Users/Jamie Phelps/Documents/RubiksNew/Datasets/MCTS/ds.pkl', 'rb') as file:
        data = pickle.load(file)
    indexes = np.arange(len(data))
    np.random.shuffle(indexes)
    batch = []
    for idx in indexes:
        batch.append(data[idx])
        if len(batch) == batch_size:
            yield batch
            batch = []


def learn(epochs, opt, batch_size=128):
    for _ in range(epochs):
        losses = []
        for batch in batch_generator(batch_size=batch_size):
            batch = np.array(batch)
            states = np.array(batch[:, 0].tolist())
            targets = np.array(batch[:, 1].tolist())

            states = np.reshape(states, (batch_size, 3 ** 3))
            idxs = np.where(states[0] == -1)[0]
            states = np.delete(states, idxs, axis=-1)
            states = tf.one_hot(states, depth=24)

            with tf.GradientTape() as tape:
                out = net(states)
                loss = tf.reduce_mean(tf.square(targets - out))

            grads = tape.gradient(target=loss, sources=net.trainable_variables)
            opt.apply_gradients(zip(grads, net.trainable_variables))
            losses.append(loss)
        print(np.mean(losses))


def MCTS_test(num_tests, depth=20, simulations=800):
    score = 0
    for _ in range(num_tests):
        print(_)
        env = RubiksEnv().reset(depth)
        j = 0
        while (j < depth) and not env.is_solved():
            print(j)
            node, actions = search(env, simulations=simulations, training=False)
            if actions is not None:
                for a in actions:
                    env.take_action(a)
            else:
                a = np.argmax(node.q_values)
                env.take_action(a)
            j += 1
        if env.is_solved():
            print('e')
            score += 1
    return score / num_tests


def test(q_net, scram, tests=100):
    env = RubiksEnv()
    solved = []
    for i in range(tests):
        print(i)
        env.reset(scram)
        total = 0
        done = 0
        while total < 2 * scram:

            state = np.reshape(env.state(), (1, 3 ** 3))
            idxs = np.where(state[0] == -1)[0]
            state = np.delete(state, idxs, axis=-1)
            state = tf.one_hot(state, depth=24)
            # state = tf.one_hot(np.array([env.state()]), depth=24)
            action = np.argmax(q_net(state))
            env.take_action(action)
            if env.is_solved():
                done = 1
                break
            total += 1
        solved.append(done)
    print(f'test{scram}: {np.mean(solved)}')


if __name__ == '__main__':
    inp = tf.keras.layers.Input((20, 24))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(12)(x)

    net = tf.keras.models.Model(inp, x)

    net.load_weights('/home/jamie/IdeaProjects/RubiksNew/DQN/FC_q.h5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    net = tf.saved_model.load("/home/jamie/IdeaProjects/RubiksNew/DQN/savedmodel/")


    # test(net, 15, tests=500)

    env = RubiksEnv()
    env.take_action(9)
    env.take_action(1)
    env.take_action(4)
    # env.take_action(1)
    # env.take_action(3)
    # env.take_action(1)
    # env.take_action(4)

    start = default_timer()
    node = search(env, 600, False)[0]
    print((default_timer() - start) * 1000000)
    print(node.q_values)
    print(node.action_counts)
    print(node.children[5].action_counts)
    exit()

    print(MCTS_test(50, simulations=600, depth=15))
    net.load_weights('C:/Users/Jamie Phelps/Documents/RubiksNew/DQN/FC_q.h5')
    print(MCTS_test(100, simulations=400, depth=20))
    exit()

    for _ in range(10000):
        print(_)
        get_dataset(2048, simulations=600)
        learn(10, optimizer, batch_size=128)
        net.save_weights('C:/Users/Jamie Phelps/Documents/RubiksNew/AlphaZero/FC_q.h5')


# env = RubiksEnv()
# env.reset(15)
# state = np.reshape(env.state(), (1, 3 ** 3))
# idxs = np.where(state[0] == -1)[0]
# state = np.delete(state, idxs, axis=-1)
# state = tf.one_hot(state, depth=24)
# print(np.array(net(state))[0])
# init, a = search(env, simulations=800)
# print(init.q_values)
# print(init.action_counts)
# print('*' * 50)
# exit()
#
# solved = 0
# for _ in range(100):
#     print(_)
#     env.reset(20)
#     j = 0
#     while (j < 30) and not env.is_solved():
#         start, action_list = search(env.copy(), simulations=1000)
#         if action_list is not None:
#             [env.take_action(a) for a in action_list]
#         else:
#             a = np.argmax(start.q_values)
#             env.take_action(a)
#         j += 1
#     if env.is_solved():
#         print('solved')
#         solved += 1
#
# print(solved)




