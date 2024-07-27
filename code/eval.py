import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from TS_RSR import TS_RSR
from Bayes_basic import ts
from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
from skopt.space import Categorical
rewards_path = '../dataset/2_chem_Buchwald_Hartwig .csv'
space_path = '../dataset/2_搜索空间.csv'
space = pd.read_csv(space_path, na_values=['NA', 'na', ''], keep_default_na=False)
rewards = pd.read_csv(rewards_path, na_values=['NA', 'na', ''], keep_default_na=False)
rewards.fillna('Non', inplace=True)
space.fillna('Non', inplace=True)

np.random.seed(37)
result_ts = []
result_bayes = []
result_rf = []
result_random = []
result_gb = []
T = 25
random_times = 5
seeds = []
for i in range(5):
    seeds.append(random.randint(0, 1000000))
    
for seed_t in range(5):
    # result_ts.append(TS_RSR(T, seeds[seed_t], random_times, space, rewards))
    # result_bayes.append(ts(T, seeds[seed_t], random_times, space, rewards, gp_minimize))
    result_random.append(ts(T, seeds[seed_t], random_times, space, rewards, dummy_minimize))
y_ts = []
for i in range(100):
    y_ts.append(np.average(result_ts[i]))
x = np.arange(random_times, T, 5)
plt.plot(x, result_ts, label='TS-RSR')
plt.xlabel('Test times')
plt.ylabel('yields')
plt.show()