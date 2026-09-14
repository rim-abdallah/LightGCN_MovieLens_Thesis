import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Settings
CSV_PATH = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\user_item_probabilities.csv"
)
USER_ID = "User 457"

# Portion to display, in the original CSV column order
START_FILE = 0
END_FILE = 200  # excluded; use None to show all files

# Read user IDs as text
df = pd.read_csv(CSV_PATH, dtype=str)
user_ids = df.iloc[:, 0].str.strip()
selected = df.loc[user_ids == str(USER_ID).strip()]

if len(selected) != 1:
    raise ValueError(
        f"Expected one row for user '{USER_ID}', found {len(selected)}. "
        f"Example user IDs: {user_ids.head().tolist()}"
    )

# File IDs come directly from the column headers
file_ids = df.columns[1:].to_numpy()
probabilities = pd.to_numeric(
    selected.iloc[0, 1:], errors="raise"
).to_numpy(dtype=float)

if (
    not np.isfinite(probabilities).all()
    or (probabilities < 0).any()
    or (probabilities > 1).any()
):
    raise ValueError("Invalid probabilities.")

if not np.isclose(probabilities.sum(), 1.0, atol=1e-6, rtol=0):
    raise ValueError("The user's probabilities do not sum to 1.")

file_ids = file_ids[START_FILE:END_FILE]
shown = probabilities[START_FILE:END_FILE]

if len(shown) == 0:
    raise ValueError("The selected file range is empty.")

# Equally spaced files, labeled with their actual CSV IDs
x = np.arange(len(shown))
ticks = np.unique(
    np.linspace(0, len(shown) - 1, min(6, len(shown)), dtype=int)
)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(x, shown * 100, color="royalblue", linewidth=2.5)

ax.set_xticks(ticks)
ax.set_xticklabels(file_ids[ticks])
ax.set_xlabel("File ID", fontsize=22)
ax.set_ylabel("probability (%)", fontsize=22)
ax.set_title(f"{USER_ID} Distribution", fontsize=23, pad=15)
ax.tick_params(axis="both", labelsize=18)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(
    CSV_PATH.parent / "selected_user_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()