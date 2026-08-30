import pandas as pd
import os

# ==========================
# USER INPUT
# ==========================

k = int(input("Enter Top-K: "))

input_file = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\output_scores\probabilities\user_item_probabilities.xlsx"

output_file = rf"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\top_{k}_user_item_probabilities.csv"

# ==========================
# LOAD MATRIX
# ==========================

df = pd.read_excel(input_file, index_col=0)

# Check K is valid
if k <= 0:
    raise ValueError("K must be greater than 0")

if k > df.shape[1]:
    raise ValueError(
        f"K={k} is larger than the number of items ({df.shape[1]})"
    )
print(f"Users: {df.shape[0]}")
print(f"Items: {df.shape[1]}")

# ==========================
# KEEP TOP-K
# ==========================

result = pd.DataFrame(
    0.0,
    index=df.index,
    columns=df.columns
)

for user in df.index:

    user_probs = df.loc[user]

    top_k = user_probs.nlargest(k)

    # normalize so sum = 1
    normalized = top_k / top_k.sum()

    result.loc[user, normalized.index] = normalized

# ==========================
# SAVE
# ==========================

os.makedirs(os.path.dirname(output_file), exist_ok=True)

result.to_csv(output_file)

print(f"\nSaved: {output_file}")

# verify
print("\nFirst 5 row sums:")
print(result.sum(axis=1).head())