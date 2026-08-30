import os
import numpy as np
import pandas as pd

# =====================================================
# SETTINGS
# =====================================================
OUTPUT_FOLDER = r"Shared\Movie_Lens\zipf_communities"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

N_USERS_PER_COMMUNITY = 100
N_FILES = 1000
ZIPF_ALPHA = 0   # change this if you want stronger/weaker Zipf

COMM0_CSV = os.path.join(OUTPUT_FOLDER, "community_0_zipf.csv")
COMM1_CSV = os.path.join(OUTPUT_FOLDER, "community_1_zipf.csv")
MERGED_CSV = os.path.join(OUTPUT_FOLDER, "merged_200_users_zipf.csv")

# =====================================================
# ZIPF DISTRIBUTION FUNCTION
# =====================================================
def zipf_distribution(n_items, alpha=1.0):
    ranks = np.arange(1, n_items + 1)
    probs = 1.0 / (ranks ** alpha)
    probs = probs / probs.sum()
    return probs

# Zipf over 200 files only
zipf_200 = zipf_distribution(200, ZIPF_ALPHA)

# =====================================================
# COMMUNITY 0
# first 200 files Zipf, last 800 files zero
# users: User_0 to User_99
# =====================================================
comm0_probs = []

for u in range(N_USERS_PER_COMMUNITY):
    row = np.zeros(N_FILES)
    row[:200] = zipf_200
    comm0_probs.append(row)

comm0_df = pd.DataFrame(
    comm0_probs,
    columns=[f"file_{i}" for i in range(N_FILES)]
)

comm0_df.insert(0, "user", [f"User_{i}" for i in range(N_USERS_PER_COMMUNITY)])

# =====================================================
# COMMUNITY 1
# files 200-399 Zipf, all others zero
# users: User_0 to User_99 inside community CSV
# =====================================================
comm1_probs = []

for u in range(N_USERS_PER_COMMUNITY):
    row = np.zeros(N_FILES)
    row[200:400] = zipf_200   # Files 200-399
    comm1_probs.append(row)

comm1_df = pd.DataFrame(
    comm1_probs,
    columns=[f"file_{i}" for i in range(N_FILES)]
)

comm1_df.insert(
    0,
    "user",
    [f"User_{i}" for i in range(N_USERS_PER_COMMUNITY)]
)

# =====================================================
# MERGED CSV
# users: User_0 to User_199
# =====================================================
merged_df = pd.concat(
    [
        comm0_df.drop(columns=["user"]),
        comm1_df.drop(columns=["user"])
    ],
    ignore_index=True
)

merged_df.insert(0, "user", [f"User_{i}" for i in range(2 * N_USERS_PER_COMMUNITY)])

# =====================================================
# CHECK SUMS
# =====================================================
print("Community 0 row sums:")
print(comm0_df.iloc[:, 1:].sum(axis=1).head())

print("Community 1 row sums:")
print(comm1_df.iloc[:, 1:].sum(axis=1).head())

print("Merged row sums:")
print(merged_df.iloc[:, 1:].sum(axis=1).head())

# =====================================================
# SAVE
# =====================================================
comm0_df.to_csv(COMM0_CSV, index=False)
comm1_df.to_csv(COMM1_CSV, index=False)
merged_df.to_csv(MERGED_CSV, index=False)

print("Saved:")
print(COMM0_CSV)
print(COMM1_CSV)
print(MERGED_CSV)