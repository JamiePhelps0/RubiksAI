# from multiprocessing import Pool
# import pickle
# import Env
# import numpy as np
# from rubik_solver import utils
# from timeit import default_timer as timer
# from numba import jit, cuda
#
#
# actions = {'F': 0, "F'": 1, 'U': 2, "U'": 3, 'L': 4, "L'": 5, 'R': 6, "R'": 7, 'D': 8, "D'": 9, 'B': 0, "B'": 11}
#
#
# def play_game(environment):
#     environment.reset()
#     approx_solve = [str(i) for i in utils.solve(environment.cube_str(), 'Kociemba')]
#     approx_solve = [[i] if not i[-1].isdigit() else [i[:-1], i[:-1]] for i in approx_solve]
#     approx_solve = [j for i in approx_solve for j in i]
#     data = []
#     for i in range(len(approx_solve)):
#         environment.take_action(actions[approx_solve[i]])
#         value = -len(approx_solve[i:])
#         data.append([actions[approx_solve[i]], environment.state(), value])
#     return np.array(data)
#
#
# def get_dataset(num_games):
#     data = play_game(Env.RubiksEnv())
#     for i in range(num_games - 1):
#         game = play_game(Env.RubiksEnv())
#         data = np.concatenate([data, game])
#     return data
#
#
# def mp(games, processes):
#     with Pool(processes=processes) as pool:
#         results = pool.map(get_dataset, [games // processes for _ in range(processes)])
#     return results
#
#
# if __name__ == "__main__":
#     for i in range(100000):
#         data = mp(10000, 10)
#         print(i)
#         with open(f'C:/Users/Jamie Phelps/Documents/AlphaRubiks/supervisedDatasets2/data{i}.pkl', 'wb') as file:
#             pickle.dump(data, file)


from multiprocessing import Pool
from timeit import default_timer
import Env
import numpy as np
import random
import pycuber as pc
import pickle

actions = {0: "F", 1: "F'", 2: "U", 3: "U'", 4: "L", 5: "L'", 6: "R", 7: "R'", 8: "D", 9: "D'", 10: "B", 11: "B'"}
actions_p = {'F': 0, "F'": 1, 'U': 2, "U'": 3, 'L': 4, "L'": 5, 'R': 6, "R'": 7, 'D': 8, "D'": 9, 'B': 10, "B'": 11}


def play_game(environment):
    environment.cube = pc.Cube()
    data = []
    action = random.randint(0, 11)
    prev_state = environment.state()
    environment.take_action(action)
    state = environment.state()
    move = actions[action]
    move_p = move + "'" if len(move) == 1 else move[0]
    target = actions_p[move_p]
    data.append([state, target, prev_state])
    for j in range(29):
        prev_state = state
        action = random.randint(0, 11)
        environment.take_action(action)
        move = actions[action]
        move_p = move + "'" if len(move) == 1 else move[0]
        target = actions_p[move_p]
        state = environment.state()
        data.append([state, target, prev_state])
    return np.array(data)


def get_dataset(num_games):
    data = play_game(Env.RubiksEnv())
    for i in range(num_games - 1):
        game = play_game(Env.RubiksEnv())
        data = np.concatenate([data, game])
    return data


def mp(games, processes):
    with Pool(processes=processes) as pool:
        results = pool.map(get_dataset, [games // processes for _ in range(processes)])
    return results


if __name__ == "__main__":
    play_game(Env.RubiksEnv())
    for i in range(10):
        time = default_timer()
        data = np.array(mp(500_000, 10))
        data = np.vstack(data)
        # _, idxs = np.unique(np.array(data[:, 1].tolist()), return_index=True, axis=0)
        # print(idxs, idxs.shape, data.shape)
        # data = data[idxs]
        print(i)
        with open(f'C:/Users/Jamie Phelps/Documents/RubiksNew/Datasets/3x3x3_State_DS/data{i}.pkl', 'wb') as file:
            pickle.dump(data, file)
    #     print(default_timer() - time)





