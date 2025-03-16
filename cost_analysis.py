import pandas as pd
import tiktoken
from tqdm import tqdm

# Load CSV
csv_file = "data/sampled Exact and Earth Sciences_Chemistry abstracts.csv"
df = pd.read_csv(csv_file, decimal='.', sep=',', encoding='utf-8')

# Remove NaN values
df['abstract'] = df['abstract'].fillna("")

# Define model prices
model_data = {
    "gpt-4.5-preview-2025-02-27": {"encoding": "cl100k_base", "price_input": 75.0, "price_output": 150.0},
    "deepseek-chat": {"encoding": None, "price_input": 0.27, "price_output": 1.10},
    "gemini-2.0-flash-thinking-exp": {"encoding": "cl100k_base", "price_input": 0.0, "price_output": 0.0},
    "Grok": {"encoding": "cl100k_base", "price_input": 2.0, "price_output": 10.0},
    "Mistral": {"encoding": "cl100k_base", "price_input": 0.0, "price_output": 0.0},
}

# Function to count tokens
def count_tokens(text):
    """Counts tokens using cl100k_base encoding."""
    if not isinstance(text, str) or not text.strip():
        return 0  # Returns zero for invalid values

    encoding_name = "cl100k_base"

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error encoding text: {e}")
        return 0

    return 0  # Returns zero if encoding is not defined

# Apply token counting
tqdm.pandas(desc="Calculating tokens")
df['abstract_tokens'] = df.progress_apply(lambda row: count_tokens(row['abstract']), axis=1)

# Create the 'results_tokens' column if it doesn't exist
df['results_tokens'] = 2 * df['abstract_tokens']

# Create the cost table for ALL models
data = []
for model in model_data.keys():
    
    # If the model exists in the DataFrame, use the real values
    if not df.empty:
        tokens_input = df['abstract_tokens'].sum() / 1_000_000
        tokens_output = df['results_tokens'].sum() / 1_000_000
    else:
        # Otherwise, set values to zero
        tokens_input = 0
        tokens_output = 0
    
    # Model prices
    price_input = model_data[model]["price_input"]
    price_output = model_data[model]["price_output"]

    # Calculate total cost considering input and output
    total_cost = (tokens_input * price_input) + (tokens_output * price_output)

    data.append([
        model,
        tokens_input + tokens_output,
        total_cost
    ])

# Create DataFrame of results
results_df = pd.DataFrame(data, columns=[
    "Model",
    "Total Tokens (M)",
    "Total Cost ($)"
])

# Display the results
print("\n🔹 Costs (Dollars) and Tokens (Millions) per Model:")
print(results_df.to_string(index=False))

# Calculate totals
total_tokens = results_df["Total Tokens (M)"].sum()
total_cost_all_models = results_df["Total Cost ($)"].sum()

print("\n🔹 Totals:")
print(f"  - Total Tokens: {total_tokens:.3f}M")
print(f"  - Total Cost (All Models): ${total_cost_all_models:.2f}")
