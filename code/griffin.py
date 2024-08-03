import numpy as np
import pandas as pd
from gryffin import Gryffin
rewards_path = '../dataset/2_chem_Buchwald_Hartwig .csv'
space_path = '../dataset/2_搜索空间.csv'
space = pd.read_csv(space_path, na_values=['NA', 'na'], keep_default_na=False)
rewards = pd.read_csv(rewards_path, na_values=['NA', 'na'], keep_default_na=False)

# 清洗数据
space.fillna('', inplace=True)
rewards.fillna(np.nan, inplace=True)

# 定义配置空间
def create_configuration_space(space):
    # print(space)
    parameters = []
    for i in range(len(space.iloc[0])):
        parameters.append([])
        for j in range(len(space.iloc[1:,i])):
            if space.iloc[j,i] != '':
                parameters[i].append(space.iloc[j,i])
    orz = dict(zip(space.columns, parameters))
    return orz
cs = create_configuration_space(space) 
def objective_function(config, seed=None):
    params = []
    global result
    for key, value in config.items():
        params.append(value)
    print(params)
    length = len(params)
    for i in range(len(rewards.iloc[1:,0])):
        isin = 0
        for j in range(len(params)):
            if(params[j] == np.nan):
                length -= 1
            for k in range(len(rewards.iloc[0])):
                if rewards.iloc[i, k] == params[j]:
                    isin=isin+1
            if isin == length:
                result.append(-rewards.iloc[i,len(rewards.iloc[i])-1])
                return -rewards.iloc[i,len(rewards.iloc[i])-1]
    result.append(0)
    return 0
config = {
    "parameters":[cs],
    objectives:[{"name":"obj", "goal":"min"}],
}
gryffin= Gryffin(config_dict=config)
observations = []
result = []
for _ in range(100):
    params = gryffin.recommend(observations=observations)
    merit = objective_function(params["parameters"])
    params['obj'] = merit
    observations.append(params)
    result.append(merit)
