import pandas as pd
df = pd.read_csv(r"./capo_prompt.csv")
        #set prompt for every module

meta_plan_prompt = df["prompt"][0]
host_prompt = df["prompt"][1]
teammate_prompt = df["prompt"][2]
refiner_prompt = df["prompt"][3]
parsing_prompt = df["prompt"][4]
print(meta_plan_prompt)
# print(host_prompt)
# print(teammate_prompt)
# print(refiner_prompt)
# print(parsing_prompt)