import re
import pandas as pd

df_list = []
try:
    with open("data/Exact and Earth Sciences_Chemistry.txt", 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r'Abstract:\s*(.*)', line, re.IGNORECASE)
            if match:
                df_list.append(match.group(1).strip())
except FileNotFoundError:
    print("Error: File not found at data/Exact and Earth Sciences_Chemistry.txt")

df = pd.DataFrame({'abstract': df_list})

df['nchar'] = df['abstract'].apply(len)
df = df[df['nchar'] > 20]

print(df['nchar'].describe())

print(df.head())


sampled_df = df['abstract'].sample(n=89, random_state=42)
sampled_df.to_csv("data/sampled Exact and Earth Sciences_Chemistry abstracts.csv", index=False)
