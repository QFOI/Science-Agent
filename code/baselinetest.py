import httpx
import pandas as pd
import numpy as np
from openai import OpenAI
rewards_path = '../dataset/2_chem_Buchwald_Hartwig .csv'
space_path = '../dataset/2_搜索空间.csv'
space = pd.read_csv(space_path, na_values=['NA', 'na', ''], keep_default_na=False)
rewards = pd.read_csv(rewards_path, na_values=['NA', 'na', ''], keep_default_na=False)
rewards.fillna('Non', inplace=True)
space.fillna('Non', inplace=True)
client = OpenAI(
    base_url="https://api.xty.app/v1", 
    api_key="sk-e3Qbj2weTQ50AACYEb3c279709B94e2bA115F190C96100B9",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)
space = space.to_dict('list')
def main():
    max_value = 0
    def get_reward(paramsquence):
      # print(params)
      length = len(paramsquence)
      for i in range(len(rewards.iloc[1:,0])):
          isin = 0
          for j in range(len(paramsquence)):
              if(paramsquence[j] == 'Non'):
                  length -= 1
              for k in range(len(rewards.iloc[0])):
                  if rewards.iloc[i, k] == paramsquence[j]:
                      isin=isin+1
              if isin == length:
                  return rewards.iloc[i,len(rewards.iloc[i])-1]/100
      return 0
    num_rounds = 20 
    question = f"the yield of synthesis Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1"
    T = 5
    completion_history = [
      {"role": "system", "content": f"You are a PhD student, you are trying to maximize {question}, You need to choose one value in different variable, the variable are give by Python's dictionary type, which key is the group's name, and value is a list of parameters in each group. The value space of each variable are follow: {space}, do not choose 'Non'\n"}
    ]
    return_yield = 0
    for _ in range(num_rounds):
        print(_)
        if _ == 0:
            completion_history.append({"role": "user", "content": 
            f"considering the properties of each possible value in each variable and their \
            contribution to the question and based on your understanding of the question, \
            give me {T} solutions, you only need to return the variable' name you choosed, \
            and use @ between different solution, use $ between value of different variable. For exmaple, you only need to response \
            [the name of your choosed value of first variable in solution 1 $ ...] @ ..."})
        completion = client.chat.completions.create(
          model="gpt-4o",
          messages=completion_history
          # {} {} {}
        )
        params = completion.choices[0].message.content
        params = params.replace(" ", "")
        params = params.replace("'", "")
        params = params.split("@")
        # print(params)
        experiment_result = []
        for i in range(T):
            # print(params[i])
            params[i] = params[i].replace("[","")
            params[i] = params[i].replace("]","")
            params[i] = params[i].split("$")
            return_yield = get_reward(params[i])
            max_value = max(return_yield, max_value)
            experiment_result.append(f"variable choose {params[i]}, yield is {return_yield}\n")
            print(f"{params[i]} -> {return_yield}")
        # print(experiment_result)
        completion_history.append({"role": "user", "content": f"Here are the results from the experiments:\n{experiment_result}\n \
            Reflect on the experimental results, considering the properties of each possible value in each variable and their \
            contribution to the question:{question} and try to explore the less explored value of variable in the value space:{space}\n \
            With previous experiment results, please continue to design solutions to achieve higher yield. You should only response \
            the variable' name you choosed, and use @ between different solution, use $ between value of different variable. For exmaple, you only need to response \
            [the name of your choosed value of fisrt variable in solution 1 $ ...] @ ...\n, if the history experiment numbers is less than 5, you should response random\
        "})
        
    print(f"the highest value is {max_value}")
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