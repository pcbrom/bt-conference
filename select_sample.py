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

sampled_df = df.sample(n=89, random_state=42)
sampled_df_rep = sampled_df.loc[sampled_df.index.repeat(3)].reset_index(drop=True)
sampled_df_rep['Repetition'] = [1, 2, 3] * (len(sampled_df_rep) // 3) + [1, 2, 3][:len(sampled_df_rep) % 3]
cols_to_create = ['ZH_EN', 'EN_ZH']
for col in cols_to_create:
    sampled_df_rep[col] = ''

sampled_df_rep = sampled_df_rep.drop('nchar', axis=1)

sampled_df_rep.to_csv("data/sampled Exact and Earth Sciences_Chemistry abstracts.csv", index=False)
