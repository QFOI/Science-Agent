import numpy as np
import pandas as pd
import matplotlib as plt
from smac import Scenario
from ConfigSpace import ConfigurationSpace
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOF
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade as ACF
from smac.runhistory.runhistory import RunHistory
from smac.runhistory import dataclasses
from smac.initial_design.random_design import RandomInitialDesign
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
    cs = ConfigurationSpace(orz)
    return cs

cs = create_configuration_space(space)

# 定义情景
scenario = Scenario(
    configspace=cs,
    n_trials=300,  # 最大运行次数
    n_workers=1,  # 使用的工作者数量
    seed=-1,
)

# 定义目标函数
result = []
res = []
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
# 创建SMAC优化器

class CustomHPOF(HPOF):
    def __init__(self, scenario, target_function, initial_design=None):
        super().__init__(scenario, target_function)
        if initial_design is not None:
            self._initial_design = initial_design

# 创建自定义的初始设计


# 运行优化并保存结果


    # 回调函数，保存每次迭代的结果
    #def save_results(incumbent, *args, **kwargs):
    #    additional_info = kwargs.get('additional_info', {})
    #    result.append((incumbent.config, -incumbent.additional_run_info['quality']))

    # 运行优化
    # for k in range(smac.runhistory.__len__()):
    # result = dataclasses.TrialValue()
    
# 调用函数
def smac33():
    custom_initial_design = RandomInitialDesign(
        scenario=scenario,
        n_configs=5,
        n_configs_per_hyperparameter=5,
        max_ratio=0.20,
        additional_configs=None,
    )
    smac = CustomHPOF(
        scenario=scenario,
        target_function=objective_function,
        initial_design=custom_initial_design,
    )
    smac.optimize()
    return result
res = smac33()

def draw_func(f, x_label, arm_num):
    nums = 300
    x = np.linspace(1, nums, nums,endpoint=True, retstep=False, dtype=int, axis=0)
    y = []
    y_min = []
    y_max = []
    for __ in range(nums):
        sum = 0
        _min = 100
        _max = 0
        for _ in range(arm_num):
            sum += -f[_][__]
            _min = min(_min, -f[_][__])
            _max = max(_max, -f[_][__])
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
draw_func(res, "Smac3 Optimization", 1)