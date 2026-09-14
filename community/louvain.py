import os
import time
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
    total_start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("Loading probability CSV...")
    t0 = time.time()
    df_probs = pd.read_csv(csv_path, index_col=0)
    probs = df_probs.values
    n_users = probs.shape[0]
    print(f"Data loaded in {time.time() - t0:.2f} seconds.")

    print("Computing user-user similarity...")
    t0 = time.time()
    sim_matrix = cosine_similarity(probs)
    print(f"Similarity computed in {time.time() - t0:.2f} seconds.")


    print("Building user-user graph...")
    t0 = time.time()
    G = nx.Graph()
    G.add_nodes_from(range(n_users))

    for u in range(n_users):
        similarities = sim_matrix[u].copy()
        similarities[u] = -1

        top_neighbors = np.argsort(similarities)[::-1][:top_k_neighbors]

        for v in top_neighbors:
            sim = similarities[v]

            if sim > similarity_threshold:
                G.add_edge(u, v, weight=float(sim))

    print(f"Graph built in {time.time() - t0:.2f} seconds.")
    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())


    print("Running Louvain...")
    t0 = time.time()
    partition = community_louvain.best_partition(
        G,
        weight="weight",
        random_state=42
    )
    print(f"Louvain completed in {time.time() - t0:.2f} seconds.")

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

    print("Exporting community CSVs...")
    t0 = time.time()
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

    print(f"CSVs exported in {time.time() - t0:.2f} seconds.")

    # Total Runtime
    elapsed_time = time.time() - total_start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    return user_communities


csv_path = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\user_item_probabilities_1000.csv"

user_communities = louvain_from_probability_csv(
    csv_path=csv_path,
    top_k_neighbors=20,
    similarity_threshold=0.0,
    output_dir=r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities_2"
)