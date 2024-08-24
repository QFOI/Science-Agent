import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
rewards_path = '../dataset/2_chem_Buchwald_Hartwig .csv'
space_path = '../dataset/2_搜索空间.csv'
space = pd.read_csv(space_path, na_values=['NA', 'na', ''], keep_default_na=False)
rewards = pd.read_csv(rewards_path, na_values=['NA', 'na', ''], keep_default_na=False)
rewards.fillna('None', inplace=True)
space.fillna('', inplace=True)
client = OpenAI(
    base_url="https://api.xty.app/v1", 
    api_key="sk-40W4OvPYBsrKYEUL0bA6EcF4E55746A9B7A03f6cA15fA581",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)
space = space.to_dict('list')
def main():
    max_value = 0
    # 查询函数
    def get_reward(paramsquence):
      # print(params)
      length = len(paramsquence)
      for i in range(len(rewards.iloc[1:,0])):
          isin = 0
          for j in range(len(paramsquence)):
              if(paramsquence[j] == ''):
                  length -= 1
              for k in range(len(rewards.iloc[0])):
                  if rewards.iloc[i, k] == paramsquence[j]:
                      isin=isin+1
              if isin == length:
                  return rewards.iloc[i,len(rewards.iloc[i])-1]/100
      return 0
    num_rounds = 20
    question = f"Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1"
    T = 1
    completion_initial = [
      {"role": "system", "content": f"You are a PhD student, you try to use methyl-5-(thiophen-2-yl)isoxazole-3-carboxylate to synthesize {question},\
          and you have a list of the variable which includes compounds you can use: {space}."}
    ]
    result = []
    chat_log = []
    res = []
    for __ in range(1):
        return_yield = 0
        completion_history = completion_initial
        completion_history.append({"role": "user", "content":f"Explain each compound's property, and their possible contribution to this synthesis."})
        return_yield = []
        for _ in range(num_rounds):
            if _ == 0:
                completion_history.append({"role": "user", "content": 
                f"think how {question} to be synthesized, select one compound from each variable and explain the reason"})
            completion = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=completion_history
                # {} {} {}
                )
            params = completion.choices[0].message.content
            chat_log.append(str(f"Round:{__}, Times:{_}")+params)
            completion_history.append({"role":"assistant","content":f"{params}"})
            # print(params)
            # 另外使用一个gpt从答案解释中或许方案，用com_his记录
            com_his = completion_initial
            com_his.append({"role": "user", "content": f"pick up the selected three compounds' name from {params}, You should only response three compounds' name like ...&... with no more words, use & between two compounds."})
            com = client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=com_his
            )
            params = com.choices[0].message.content
            # print(params)
            # print(f"Round:{__}, times{_},choose:{params}")
            '''
            pick_up_compounds = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": f"pick up the compounds' name from {params}."}
                ]
            )
            params = pick_up_compounds.choices[0].message.content
            print(params)
            '''
            params = params.replace(" ", "")
            params = params.replace("'", "")
            # params = params.replace("[", "")
            # params = params.replace("]", "")
            params = params.replace("*", "")
            params = params.split("&")
            # 三变量搜索
            params.append("methyl-5-(thiophen-2-yl)isoxazole-3-carboxylate")
            experiment_result = []
            for i in range(1):
                # print(params[i])
                #params[i] = params[i].replace("[","")
                #params[i] = params[i].replace("]","")
                #params[i] = params[i].split("$")
                return_yield.append(get_reward(params))
                max_value = max(return_yield[_], max_value)
                # experiment_result.append(f"variable choose {params[i]}, yield is {return_yield}\n")
                print(f"Round:{__}, times{_},choose:{params} -> {return_yield[_]}")
            # print(experiment_result)
            completion_history = [{"role": "user", "content": f"the field is {return_yield[_]}, \
                please continue to enhance the yield by optimizing your plan. \
                If you think it's probably near maximum yield, try to find some unexplored combinations with less explored compounds. \
                and explain your reason.\
                All compounds' name is from {space}."}]
        tot = 0
        maxt = 0
        res.append(return_yield)
        for ___ in range(len(return_yield)):
            tot += return_yield[___]
            maxt = max(maxt, return_yield[___])
        for ___ in range(len(return_yield)):
            return_yield[___] = (return_yield[___]-tot/50)/(maxt-tot/50)
        result.append(return_yield)
        print(f"Round{__},the highest value is {max_value}")
    # draw_func(result, "Single Agent 2 Prompts", 10)
    # draw_func(res, "Single Agent 2 Prompts Yield", 10)
    chat_log = np.array(chat_log)
    np.savetxt('123.txt', chat_log, fmt='%s')
    return res, result
if __name__ == "__main__":
    main()
'''
completion = client.chat.completions.create(
  model="gpt-4o-mini-2024-07-18",
  messages=[
    {"role": "system", "content": "帮我检索构建了MOFs材料相关属性数据集的文献，并检索用这些数据集微调过的具有MOfs材料专业性能的大语言模型,选出在该领域相关问题上回答相应最好或者知识最丰富的模型，并帮我写一个通过huggingface引用该模型的python代码，并填入查询的问题“什么MOFs材料的CO2通过率最高”，注意，只需要返回你所提供的代码"}
  ]
)
print(completion.choices[0].message.content)
'''