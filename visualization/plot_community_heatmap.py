import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def clean_user_id(value):
    value = str(value).strip()
    value = value.replace("User_", "").replace("user_", "")
    value = value.replace("User", "").replace("user", "").strip()

    if re.fullmatch(r"\d+\.0", value):
        value = value.split(".")[0]

    return value


def plot_community_heatmap(
    probabilities_file="output_scores/probabilities/user_item_probabilities.xlsx",
    communities_file="output_scores/louvain_communities/user_louvain_communities.xlsx",
    output_path="output_scores/heatmaps/user_item_probability_heatmap_by_community.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    probs_df = pd.read_excel(probabilities_file, index_col=0)
    communities_df = pd.read_excel(communities_file)

    user_col = "User_ID"
    community_col = "Community"

    probs_df.index = probs_df.index.map(clean_user_id)
    communities_df[user_col] = communities_df[user_col].map(clean_user_id)

    communities_df = communities_df[
        communities_df[user_col].isin(probs_df.index)
    ].copy()

    communities_df = communities_df.sort_values([community_col, user_col])

    sorted_users = communities_df[user_col].tolist()
    sorted_probs = probs_df.loc[sorted_users]

    print(f"Users shown: {sorted_probs.shape[0]}")
    print(f"Items shown: {sorted_probs.shape[1]}")
    print("Communities found:", sorted(communities_df[community_col].unique()))

    values = sorted_probs.values.astype(float)

    # Log scale needs positive values
    values = values + 1e-12

    vmin = np.percentile(values, 1)
    vmax = np.percentile(values, 99.9)

    plt.figure(figsize=(35, 80))

    plt.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    plt.colorbar(label="P(item | user) - log scale")
    plt.xlabel(f"All Items ({sorted_probs.shape[1]})")
    plt.ylabel(f"All Users ({sorted_probs.shape[0]}) sorted by community")
    plt.title("User-Item Probability Heatmap Sorted by Louvain Community")

    communities = communities_df[community_col].values
    changes = np.where(communities[:-1] != communities[1:])[0]

    for idx in changes:
        plt.axhline(idx + 0.5, color="white", linewidth=2)

    plt.yticks(
        ticks=np.arange(len(sorted_users)),
        labels=sorted_users,
        fontsize=2
    )

    plt.xticks(fontsize=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_path}")
    print(f"Log color scale vmin={vmin}, vmax={vmax}")


if __name__ == "__main__":
    plot_community_heatmap()