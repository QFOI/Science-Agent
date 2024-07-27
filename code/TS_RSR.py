import pandas as pd
import numpy as np
# rewards_path = '../data_table.csv'
# space_path = '../AI抗生素合成的搜索空间数据集.csv'
# space = pd.read_csv(space_path, na_values=['NA', 'na', ''], keep_default_na=False)
# rewards = pd.read_csv(rewards_path, na_values=['NA', 'na', ''], keep_default_na=False)
# rewards.fillna('Non', inplace=True)
# space.fillna('Non', inplace=True)
# save_space = space

# rewards = pd.get_dummies(rewards)
# space = pd.get_dummies(space)
 
# space = space.dropna(na_values=na_value)

def TS_RSR(T, seed, random_times, space, rewards):
    def get_reward(params):
        # print(params)
        length = len(params)
        for i in range(len(rewards.iloc[1:,0])):
            isin = 0
            for j in range(len(params)):
                if(params[j] == 'None'):
                    length -= 1
                for k in range(len(rewards.iloc[0])):
                    if rewards.iloc[i, k] == params[j]:
                        isin=isin+1
                if isin == length:
                    return -rewards.iloc[i,len(rewards.iloc[i])-1]
        return 0
    
    reward_params = []
    history_params = []
    result_record = []
    np.random.seed(seed)

    reward_params = np.array(reward_params)
    # history_params = np.array(history_params)
    result = 0

    def Hamming_Kernel(a, b, matern_alpha = 1, matern_lambda = 1):
        d = 2
        if a == b:
            d = 0
        k = matern_alpha*(1+np.sqrt(5)*d/matern_lambda+5*d**2/(3*matern_lambda**2))*np.exp(-np.sqrt(5)*d/matern_lambda)   
        return k

    for t in range(T):
        length = t

        x = []
        for i in range(len(space.iloc[0])):
            i_params_rewards = 10000
            if t in  range(0,random_times):
                pos = -1
                # print(space.iloc[14,0]=='Non')
                while pos == -1 or space.iloc[pos, i] == 'Non':
                    pos = np.random.randint(0, len(space.iloc[:,0])-1)
                x.append(space.iloc[pos, i])
                continue 
            xs = 0
            for j in range(len(space.iloc[:,0])):
                if space.iloc[j, i] == 'Non':
                    continue
                kt = np.zeros((length ,1))

                for k in range(length):
                    kt[k] = Hamming_Kernel(space.iloc[j, i], history_params[k][i])

                ckt = np.zeros((length ,length))
                for l in range(length):
                    for k in range(length):
                        ckt[l][k] = Hamming_Kernel(history_params[k-1][i], history_params[l-1][i])
                # ckt = [[Hamming_Kernel(history_params[i][k], history_params[i][l]) for k in range(length)] for l in range(length) ]
                ckt = np.array(ckt)
                sigma = np.std(reward_params)
                yt = [[ np.random.normal(0, sigma*sigma) + reward_params[k] ]for k in range(length)]

                mu = kt.T @ np.linalg.inv( ckt+sigma*sigma*np.eye(length) ) @ yt
                sigma_self = 1-kt.T @ np.linalg.inv( ckt+sigma*sigma*np.eye(length) ) @ kt
                update = (result - mu) / sigma_self
                if update < i_params_rewards:
                    i_params_rewards = update
                    xs = j
            x.append(space.iloc[xs, i])
        y = get_reward(x)
        result = min(result, y)
        reward_params = np.append(reward_params, y)
        # for i in range(len(space.iloc[0])):
        #    history_params[i] = np.append(history_params[i] ,x[i])
        history_params = history_params + [x]
        result_record.append(-result)
        print(f'test times No.{t} , the highest yield is {-result}%, new choice yield is {-y}%, selected{x}')
    return result_record