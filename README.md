# Science-Agent

eval.py中继承了skopt中的random forest，高斯代理模型的贝叶斯优化和TS-RSR（基于2024.5哈佛最新发的数学优化算法的贝叶斯优化，不过自己复现的可能出锅了）

其中读图以''填充dataframe的空白处，同时注意不要将‘None’记作空白值


其中get_reward函数为根据所选变量进行反馈值，其输入为所选变量集合


比如：搜索空间为:{"A":['1','2'],"B":['3','4']}向get_reward函数中输入的就是['1','3']（随机选的一组，因为调用的不同模型的录入方式不同，所以不需要考虑的变量顺序）


其实现为暴力搜索（故需要大量时间），后续可改成索引或者二分搜索之类的。


图像输出为平行10组100次实验（经过今天讨论50次可能就够了），记录每次平行实验结果的平均值和最大最小值绘图


gryffin只能在mac或linux环境下使用


two_prompts.py中result存的是类似coscientist的归一化结果，res存的是产量，在run_two_prompts.py中运行（否则每一轮查询之间不独立），所用的搜索空间搜索空间需要删去最后列，仅查询前三个变量
