import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time
import anthropic


# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Access and store the environment variable
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=claude_api_key)
model = 'claude-3-7-sonnet-20250219'


# Import
csv_file = "data/sampled Exact and Earth Sciences_Chemistry abstracts.csv"
df = pd.read_csv(csv_file, decimal='.', sep=',', encoding='utf-8')
# df = df.head(2)
print(df)

# Ensure the columns are explicitly set as object (string)
df['ZH_EN'] = df['ZH_EN'].astype(object)
df['EN_ZH'] = df['EN_ZH'].astype(object)

# Translate ZH -> EN and then EN -> ZH in one loop
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Translations"):
    if index % 50 == 0 and index != 0:
        print("Pausing for API rate limit...")
        time.sleep(60)

    try:
        # Step 1: Translate from ZH to EN
        prompt_zh_en = f"Translate the following Chinese text to English, providing only the translated text without any additional explanations or context: {row['abstract']}"
        response_zh_en = client.messages.create(model=model, max_tokens=1000, messages=[{"role": "user", "content": prompt_zh_en}])
        generated_text_zh_en = response_zh_en.content[0].text
        df.loc[index, 'ZH_EN'] = generated_text_zh_en

        # Step 2: Translate back from EN to ZH
        prompt_en_zh = f"Translate the following English text to Chinese, providing only the translated text without any additional explanations or context: {generated_text_zh_en}"
        response_en_zh = client.messages.create(model=model, max_tokens=1000, messages=[{"role": "user", "content": prompt_en_zh}])
        generated_text_en_zh = response_en_zh.content[0].text
        df.loc[index, 'EN_ZH'] = generated_text_en_zh

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        df.loc[index, 'ZH_EN'] = f"Error: {e}"
        df.loc[index, 'EN_ZH'] = f"Error: {e}"

# Save
output_filename = f"results_data/experimental_design_results_{model}.csv"
df.to_csv(output_filename, index=False)

print(f"Data saved to {output_filename}")
