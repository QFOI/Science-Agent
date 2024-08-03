import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
rewards_path = '../dataset/2_chem_Buchwald_Hartwig .csv'
space_path = '../dataset/2_搜索空间.csv'
space = pd.read_csv(space_path, na_values=['NA', 'na'], keep_default_na=False)
rewards = pd.read_csv(rewards_path, na_values=['NA', 'na'], keep_default_na=False)

# 清洗数据
space.fillna('', inplace=True)
rewards.fillna(np.nan, inplace=True)
def objective_function(params):
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
                return rewards.iloc[i,len(rewards.iloc[i])-1]
    return 0
def objective(trial):
    params = []
    parameters = []
    for i in range(len(space.iloc[0])):
        parameters.append([])
        for j in range(len(space.iloc[1:,i])):
            if space.iloc[j,i] != '':
                parameters[i].append(space.iloc[j,i])
    for i in range(len(space.iloc[0])):
        params.append(trial.suggest_categorical(space.columns[i], parameters[i]))
    return objective_function(params)
res = []
result = []
def report_intermediate_values(study, trial):
    global result
    result.append(trial.value)
for _ in range(10):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[report_intermediate_values])
    res.append(result)
    result = []

def draw_func(f, x_label, arm_num):
    nums = 100
    x = np.linspace(1, nums, nums,endpoint=True, retstep=False, dtype=int, axis=0)
    y = []
    y_min = []
    y_max = []
    for __ in range(nums):
        sum = 0
        _min = 100
        _max = 0
        for _ in range(arm_num):
            sum += f[_][__]
            _min = min(_min, f[_][__])
            _max = max(_max, f[_][__])
        sum = sum/arm_num
        sum = sum/100
        _min = _min/100
        _max = _max/100
        y.append(sum)
        y_min.append(_min)
        y_max.append(_max)
    plt.xticks(np.arange(1, nums+1, 1))
    plt.plot(x, y, label='')
    plt.fill_between(x, y_min, y_max, color='gray', alpha=0.2, label='Range')
    plt.title(x_label)
    plt.xlabel('Times')
    plt.ylabel('Fields')
    plt.legend()
    plt.savefig(f'{x_label}.png', dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
print(res)
draw_func(res, 'Optuna Optimization', 10)