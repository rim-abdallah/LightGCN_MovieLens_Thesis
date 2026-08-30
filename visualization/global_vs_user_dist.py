import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# INPUT
# ==========================
csv_path = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\communities_csv\community_3_users_probabilities_alpha_1.csv"
output_path = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\communities_csv\users_global_original_order_all.png"

SMOOTH_WINDOW = 15      # Moving average window
CURVE_SCALE = 20        # Height of curves
GAP = 1.0               # Space between users

# ==========================
# READ CSV
# ==========================
df = pd.read_csv(csv_path)

user_names = df.iloc[:, 0].astype(str)
P = df.iloc[:, 1:].astype(float)

# Remove negative values if any
P[P < 0] = 0

# ==========================
# GLOBAL DISTRIBUTION
# (NO SORTING)
# ==========================
P_global = P.mean(axis=0)

# Add global as first row
plot_data = pd.concat(
    [
        pd.DataFrame([P_global.values], columns=P.columns),
        P.reset_index(drop=True)
    ],
    ignore_index=True
)

labels = ["Global"] + list(user_names)

# ==========================
# OPTIONAL SMOOTHING
# ==========================
plot_data = (
    plot_data.T
    .rolling(window=SMOOTH_WINDOW,
             center=True,
             min_periods=1)
    .mean()
    .T
)

# ==========================
# SAME SCALE FOR EVERY USER
# ==========================
max_value = plot_data.values.max()

x = np.arange(1, plot_data.shape[1] + 1)

plt.figure(figsize=(18,40))

for i in range(plot_data.shape[0]):

    y = plot_data.iloc[i].values

    if max_value > 0:
        y = y / max_value

    offset = i * GAP

    plt.plot(
        x,
        offset - CURVE_SCALE * y,
        color="black",
        linewidth=0.8
    )

# ==========================
# AXES
# ==========================
plt.gca().invert_yaxis()

plt.xlabel("Items (1 - 1000)")
plt.ylabel("Users")



# show all users on y-axis
plt.yticks(
    np.arange(0, len(labels)) * GAP,
    labels,
    fontsize=6
)

plt.xlim(1, len(P.columns))
plt.title("Global Distribution and Individual User Distributions")

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print("Saved:", output_path)