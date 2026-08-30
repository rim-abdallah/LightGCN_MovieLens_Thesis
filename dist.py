import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# INPUT / OUTPUT
# =====================================================
INPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\com_2\community_2_users_probabilities.csv"
OUTPUT_FOLDER = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\plots"

K_USERS = 70
TOP_N = 200

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# READ CSV
# =====================================================
df = pd.read_csv(INPUT_CSV)

user_names = df.iloc[:, 0].astype(str)
P = df.iloc[:, 1:].astype(float)

print("Number of users:", P.shape[0])
print("Number of items:", P.shape[1])

# =====================================================
# GLOBAL POPULARITY + SORT ITEMS
# =====================================================
P_global = P.mean(axis=0)
P_global = P_global / P_global.sum()

# Sort all items by global popularity descending
sorted_items = P_global.sort_values(ascending=False).index.tolist()

# Put top 200 first, then the rest
top_items = sorted_items[:TOP_N]
remaining_items = sorted_items[TOP_N:]
new_order = top_items + remaining_items

# Reorder probability table
P = P[new_order]

# Recompute global after reordering
P_global = P.mean(axis=0)
P_global = P_global / P_global.sum()

item_names = list(P.columns)

# =====================================================
# PLOT
# =====================================================
plt.figure(figsize=(22, 12))

SCALE = 20
GAP = 1.2

k = min(K_USERS, len(P))

# ----- Global -----
offset = (k + 1) * GAP
plt.plot(P_global.values * SCALE + offset, linewidth=2.5)
plt.text(-25, offset, "Global", fontsize=11, va="center")

# ----- Users -----
for i in range(k):
    offset = (k - i) * GAP
    plt.plot(P.iloc[i].values * SCALE + offset, linewidth=1)
    plt.text(-25, offset, user_names.iloc[i], fontsize=9, va="center")

# =====================================================
# MARK TOP 200 REGION
# =====================================================
y_min = -4
y_max = (k + 2) * GAP

plt.axvspan(0, TOP_N - 1, alpha=0.12)

plt.axvline(
    x=TOP_N - 1,
    color="black",
    linestyle="--",
    linewidth=1.2
)

plt.text(
    TOP_N / 2,
    y_max - 0.5,
    f"Top {TOP_N} global items",
    ha="center",
    fontsize=11
)

# Label some top items
LABEL_EVERY = 10

for idx in range(TOP_N):
    if idx % LABEL_EVERY == 0:
        plt.text(
            idx,
            y_min - 0.2,
            item_names[idx],
            rotation=90,
            fontsize=7,
            ha="center",
            va="top"
        )

# =====================================================
# FIGURE SETTINGS
# =====================================================
plt.xlabel("Items sorted by global popularity: Top 200 first")
plt.title(f"Global Distribution and First {k} Users - Top {TOP_N} Global Items First")

plt.yticks([])
plt.grid(axis="x", alpha=0.3)

plt.xlim(0, P.shape[1] - 1)
plt.ylim(y_min - 3, y_max)

plt.tight_layout()

# =====================================================
# SAVE
# =====================================================
output_path = os.path.join(
    OUTPUT_FOLDER,
    f"global_and_first_{k}_users_top_{TOP_N}_first.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("Saved to:", output_path)