import pandas as pd
import numpy as np
from skopt import gp_minimize, dummy_minimize
from skopt.space import Categorical

def ts(T, seed, random_times, space, rewards, f):
    def get_reward(params):
        # print(params)
        length = len(params)
        for i in range(len(rewards.iloc[1:,0])):
            isin = 0
            for j in range(len(params)):
                if(params[j] == 'Non'):
                    length -= 1
                for k in range(len(rewards.iloc[0])):
                    if rewards.iloc[i, k] == params[j]:
                        isin=isin+1
                if isin == length:
                    return -rewards.iloc[i,len(rewards.iloc[i])-1]
        return 0
    parameters = []
    for i in range(len(space.iloc[0])):
        parameters.append([])
        for j in range(len(space.iloc[1:,i])):
            if space.iloc[j,i] != 'Non':
                parameters[i].append(space.iloc[j,i])
    search_space = [
        Categorical(parameters[i], name = str(space.iloc[0, i])) for i in range(len(space.iloc[0]))
    ]

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    result = []
    for i in range(T):
        if f == dummy_minimize:
            res = f(get_reward, search_space, random_state = random_state, n_calls = T, verbose = True)
        else:
            res = f(get_reward, search_space, random_state = random_state, n_calls = T, n_random_starts = random_times, verbose = True)
        result.append(-res.fun)
    return result
