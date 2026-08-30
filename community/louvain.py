import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain


def clear_old_excel_files(folder):
    os.makedirs(folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".xlsx"):
            os.remove(os.path.join(folder, file))


def louvain_from_probability_csv(
    csv_path,
    top_k_neighbors=20,
    similarity_threshold=0.0,
    output_dir="output_scores/louvain_communities"
):
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV: users as rows, items as columns
    df_probs = pd.read_csv(csv_path, index_col=0)

    # Convert probabilities to numpy array
    probs = df_probs.values

    n_users = probs.shape[0]

    print("Computing user-user similarity...")
    sim_matrix = cosine_similarity(probs)

    G = nx.Graph()
    G.add_nodes_from(range(n_users))

    print("Building user-user graph...")

    for u in range(n_users):
        similarities = sim_matrix[u].copy()
        similarities[u] = -1

        top_neighbors = np.argsort(similarities)[::-1][:top_k_neighbors]

        for v in top_neighbors:
            sim = similarities[v]

            if sim > similarity_threshold:
                G.add_edge(u, v, weight=float(sim))

    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())

    print("Running Louvain...")
    partition = community_louvain.best_partition(
        G,
        weight="weight",
        random_state=42
    )

    user_communities = np.array([partition[u] for u in range(n_users)])

    df_result = pd.DataFrame({
        "User_ID": df_probs.index,
        "Community": user_communities
    })

    save_path = os.path.join(output_dir, "user_louvain_communities.xlsx")
    df_result.to_excel(save_path, index=False)

    print("Saved Louvain communities to:", save_path)
    print("Number of communities:", len(np.unique(user_communities)))
    print(df_result["Community"].value_counts().sort_index())
    
        # Save one CSV file per community with user probabilities
    communities_csv_dir = os.path.join(output_dir, "communities_csv")
    os.makedirs(communities_csv_dir, exist_ok=True)

    df_probs_with_comm = df_probs.copy()
    df_probs_with_comm["Community"] = user_communities

    for comm_id, group in df_probs_with_comm.groupby("Community"):
        group_probs_only = group.drop(columns=["Community"])

        csv_save_path = os.path.join(
            communities_csv_dir,
            f"community_{comm_id}_users_probabilities.csv"
        )

        group_probs_only.to_csv(csv_save_path)

        print(f"Saved community {comm_id} CSV: {csv_save_path}")

    return user_communities


# Example call
csv_path = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\user_item_probabilities_1000.csv"

user_communities = louvain_from_probability_csv(
    csv_path=csv_path,
    top_k_neighbors=20,
    similarity_threshold=0.0,
    output_dir=r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities"
)