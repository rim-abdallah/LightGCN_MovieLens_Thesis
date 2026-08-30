import os
import pandas as pd


# =====================================
# INPUT
# =====================================
INPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_2_users_probabilities.csv"

OUTPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\communities_csv\community_2_users_probabilities_renamed.csv"


# =====================================
# READ CSV
# =====================================
df = pd.read_csv(INPUT_CSV, index_col=0)

print("Number of users:", len(df))


# =====================================
# RENAME USERS
# =====================================
df.index = [f"User_{i}" for i in range(len(df))]


# =====================================
# SAVE
# =====================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df.to_csv(OUTPUT_CSV)

print("Saved to:")
print(OUTPUT_CSV)