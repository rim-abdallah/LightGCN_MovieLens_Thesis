import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def visualize_louvain_communities(
    probs,
    user_communities,
    top_k_neighbors=20,
    max_users=200,
    output_dir="output_scores/graphs",
    filename="louvain_communities_visualization.png"
):
    os.makedirs(output_dir, exist_ok=True)

    n_users = probs.shape[0]

    # To avoid drawing too many users
    selected_users = np.arange(min(n_users, max_users))

    selected_probs = probs[selected_users]
    selected_communities = user_communities[selected_users]

    print("Computing similarity for visualization...")
    sim_matrix = cosine_similarity(selected_probs)

    G = nx.Graph()
    G.add_nodes_from(selected_users)

    print("Building visualization graph...")

    for idx_u, u in enumerate(selected_users):
        similarities = sim_matrix[idx_u].copy()
        similarities[idx_u] = -1

        top_neighbors_idx = np.argsort(similarities)[::-1][:top_k_neighbors]

        for idx_v in top_neighbors_idx:
            v = selected_users[idx_v]
            sim = similarities[idx_v]

            if sim > 0:
                G.add_edge(u, v, weight=float(sim))

    print("Drawing Louvain communities...")

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.25
    )

    node_colors = [
        selected_communities[np.where(selected_users == node)[0][0]]
        for node in G.nodes()
    ]

    plt.figure(figsize=(14, 10))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        node_size=80,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.15,
        width=0.5
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=6
    )

    plt.title("Louvain Communities from LightGCN Probability Output")
    plt.axis("off")

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Louvain visualization saved to:")
    print(save_path)
    
    
    
    '''graph louvain community visualization 
    lfiyo edge '''