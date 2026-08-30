import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

csv_path = input("Enter LightGCN probability CSV path: ").strip()
output_folder = input("Enter output folder path: ").strip()

os.makedirs(output_folder, exist_ok=True)

alpha = 0.8

df = pd.read_csv(csv_path)

non_prob_cols = ["user_id", "userid", "user", "movie_id", "item_id", "title", "unnamed: 0"]
prob_df = df.drop(columns=[c for c in df.columns if c.lower() in non_prob_cols], errors="ignore")

prob_df = prob_df.apply(pd.to_numeric, errors="coerce")
prob_df = prob_df.dropna(axis=1, how="all").fillna(0)

user_ids = df["User_ID"].values if "User_ID" in df.columns else np.arange(len(prob_df))

n_items = prob_df.shape[1]
item_ids = np.arange(n_items)

random_users = random.sample(range(len(prob_df)), 2)

def plot_distribution(x, probs, title, filename):
    plt.figure(figsize=(12, 5))
    plt.plot(x, probs, linewidth=1)
    plt.title(title)
    plt.xlabel("Item ID / Item Index")
    plt.ylabel("Probability")
    plt.grid(True)

    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches="tight")
    plt.close()

# Zipf, not sorted differently, just by item rank/index
ranks = np.arange(1, n_items + 1)
zipf_probs = 1 / (ranks ** alpha)
zipf_probs = zipf_probs / zipf_probs.sum()

plot_distribution(
    item_ids,
    zipf_probs,
    "Zipf Probability Distribution",
    "zipf_distribution.png"
)

for k, idx in enumerate(random_users, start=1):
    user_id = user_ids[idx]
    user_probs = prob_df.iloc[idx].values.astype(float)

    plot_distribution(
        item_ids,
        user_probs,
        f"LightGCN Probability Distribution - User {user_id}",
        f"lightgcn_user_{k}_distribution.png"
    )

print("Done. Figures saved in:", output_folder)