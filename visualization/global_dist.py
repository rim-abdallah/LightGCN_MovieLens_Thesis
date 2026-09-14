import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Settings
CSV_PATH = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\user_item_probabilities.csv"
)

# First 200 probability columns, in the original CSV order
START_FILE = 0
END_FILE = 200  # excluded; use None to show all files

# Read CSV: first column = user ID, remaining columns = probabilities
df = pd.read_csv(CSV_PATH)
probabilities = df.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
values = probabilities.to_numpy(dtype=float)

if values.size == 0:
    raise ValueError("The CSV contains no probability data.")

if (
    not np.isfinite(values).all()
    or (values < 0).any()
    or (values > 1).any()
):
    raise ValueError(
        "Probabilities must be finite numbers between 0 and 1."
    )

row_sums = values.sum(axis=1)

if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0):
    raise ValueError(
        "Each user's probabilities must sum to 1. "
        f"Observed sums: {row_sums.min():.6f} to {row_sums.max():.6f}"
    )

# Global distribution: equal weight for every user
global_distribution = values.mean(axis=0)

# Actual file IDs from the CSV headers
file_ids = probabilities.columns.to_numpy()[START_FILE:END_FILE]
shown = global_distribution[START_FILE:END_FILE]

if len(shown) == 0:
    raise ValueError("The selected file range is empty.")

# Same x-axis positions and labels as the user graph
x = np.arange(len(shown))
ticks = np.unique(
    np.linspace(0, len(shown) - 1, min(6, len(shown)), dtype=int)
)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(
    x,
    shown * 100,
    color="royalblue",
    linewidth=2.5,
)

ax.set_xticks(ticks)
ax.set_xticklabels(file_ids[ticks])

if len(shown) > 1:
    ax.set_xlim(0, len(shown) - 1)

ax.set_xlabel("File ID", fontsize=22)
ax.set_ylabel("Probability (%)", fontsize=22)
ax.set_title("Global Distribution", fontsize=24, pad=15)

ax.tick_params(axis="both", labelsize=18)
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax.set_ylim(bottom=0)

ax.grid(alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    CSV_PATH.parent / "global_distribution_slide.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()