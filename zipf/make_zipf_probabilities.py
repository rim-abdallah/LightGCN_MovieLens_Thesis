import numpy as np
import pandas as pd
import os

# =========================
# Parameters
# =========================
n_users = 200
n_files = 1000
alpha = 0

# Your shared folder path
output_dir = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\zipf_com\for 1000"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Generate full Zipf probabilities
# =========================
ranks = np.arange(1, n_files + 1)

zipf_probs = 1 / (ranks ** alpha)
zipf_probs = zipf_probs / zipf_probs.sum()

# Repeat same probabilities for all users
full_matrix = np.tile(zipf_probs, (n_users, 1))

full_df = pd.DataFrame(
    full_matrix,
    index=[f"User_{i}" for i in range(n_users)],
    columns=[f"file_{j}" for j in range(n_files)]
)

# Save full Zipf CSV
full_output_path = os.path.join(
    output_dir,
    "zipf_probabilities_all_200_users_1000_files.csv"
)

full_df.to_csv(full_output_path)

print("Full Zipf CSV saved:")
print(full_output_path)
print("Shape:", full_df.shape)
print("First user sum:", full_df.iloc[0].sum())

# =========================
# Ask if user wants Top-K
# =========================
do_top_k = input(
    "\nDo you want to generate Top-K Zipf probabilities? yes/no: "
).strip().lower()

if do_top_k in ["yes", "y"]:

    k_values = input(
        "Enter K values separated by commas (example: 10,20,50,100): "
    )

    k_values = [int(x.strip()) for x in k_values.split(",")]

    for k in k_values:

        if k <= 0 or k > n_files:
            print(f"Skipping invalid K={k}")
            continue

        top_k_probs = zipf_probs.copy()

        # Keep only first K files
        top_k_probs[k:] = 0

        # Renormalize
        top_k_probs /= top_k_probs.sum()

        top_k_matrix = np.tile(top_k_probs, (n_users, 1))

        top_k_df = pd.DataFrame(
            top_k_matrix,
            index=[f"User_{i}" for i in range(n_users)],
            columns=[f"file_{j}" for j in range(n_files)]
        )

        output_file = os.path.join(
            output_dir,
            f"zipf_probabilities_top_{k}.csv"
        )

        top_k_df.to_csv(output_file)

        print(
            f"Generated Top-{k} -> "
            f"{(top_k_df.iloc[0] > 0).sum()} non-zero files"
        )

else:
    print("No Top-K files generated.")