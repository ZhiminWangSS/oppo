import pandas as pd


df = pd.read_csv("./LLM/belief_rules.csv")

print(df['prompt'][0])
