import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# PATHS
# ==========================
input_file = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\communities_csv\community_1_users_probabilities.csv"
output_path = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\300\user_similarity_heatmap.png"

# ==========================
# READ CSV
# ==========================
df = pd.read_csv(input_file)

user_col = df.columns[0]
user_ids = df[user_col].values

# Probability matrix only
X = df.iloc[:, 1:].values

# ==========================
# COSINE SIMILARITY
# ==========================
sim_matrix = cosine_similarity(X)

# ==========================
# STATISTICS
# ==========================
upper_values = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]

print("Average similarity:", upper_values.mean())
print("Min similarity:", upper_values.min())
print("Max similarity:", upper_values.max())

# ==========================
# HEATMAP
# ==========================
plt.figure(figsize=(10, 8))

plt.imshow(sim_matrix[:50, :50], aspect="auto", cmap="viridis")
plt.colorbar(label="Cosine Similarity")

plt.title("User-User Similarity Heatmap")
plt.xlabel("Users")
plt.ylabel("Users")

# plt.xticks(
#     ticks=range(len(user_ids)),
#     labels=user_ids,
#     rotation=90,
#     fontsize=5
# )

# plt.yticks(
#     ticks=range(len(user_ids)),
#     labels=user_ids,
#     fontsize=5
# )

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print("Saved:", output_path)
print("Number of users:", len(user_ids))