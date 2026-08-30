import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_lightgcn_bipartite(
    adj_mat,
    n_users,
    sampled_users,
    sampled_items,
    output_dir="output_scores"
):
    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    G = nx.Graph()
    sampled_items_set = set(sampled_items)

    for u in sampled_users:
        G.add_node(f"U{u}", node_type="User")

        connected_nodes = adj_mat[u].nonzero()[1]

        for node in connected_nodes:
            if node >= n_users:
                item_id = node - n_users

                if item_id in sampled_items_set:
                    G.add_edge(f"U{u}", f"I{item_id}")

    for i in sampled_items:
        G.add_node(f"I{i}", node_type="Item")

    user_nodes = [f"U{u}" for u in sampled_users]
    item_nodes = [f"I{i}" for i in sampled_items]

    # Adjacency matrix: rows = users, columns = items
    adj_display = np.zeros(
        (len(sampled_users), len(sampled_items)),
        dtype=int
    )

    for row, u in enumerate(sampled_users):
        connected_nodes = adj_mat[u].nonzero()[1]

        for node in connected_nodes:
            if node >= n_users:
                item_id = node - n_users

                if item_id in sampled_items_set:
                    col = sampled_items.index(item_id)
                    adj_display[row, col] = 1

    adj_df = pd.DataFrame(
        adj_display,
        index=user_nodes,
        columns=item_nodes
    )

    print("\nAdjacency Matrix (Users x Items):")
    print(adj_df)

    # Graph positions
    pos = {}

    for i, user in enumerate(user_nodes):
        pos[user] = (0, -i)

    for i, item in enumerate(item_nodes):
        pos[item] = (1, -i)

    # Draw bipartite graph
    plt.figure(figsize=(14, 10))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=user_nodes,
        node_color="skyblue",
        node_size=900,
        label="Users"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=item_nodes,
        node_color="lightgreen",
        node_size=900,
        label="Items"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5,
        width=1.5
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    plt.title("Before Training: Bipartite Interaction Graph")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()

    bipartite_path = os.path.join(graph_dir, "bipartite_graph.png")
    plt.savefig(bipartite_path, dpi=300)
    plt.close()

    print(f"Saved {bipartite_path}")

    # Save adjacency matrix as image
    plt.figure(figsize=(10, 5))
    plt.imshow(adj_display, aspect="auto")

    plt.xticks(
        ticks=np.arange(len(item_nodes)),
        labels=item_nodes,
        rotation=90
    )

    plt.yticks(
        ticks=np.arange(len(user_nodes)),
        labels=user_nodes
    )

    plt.title("Adjacency Matrix: Users x Items")
    plt.xlabel("Items")
    plt.ylabel("Users")

    for r in range(adj_display.shape[0]):
        for c in range(adj_display.shape[1]):
            plt.text(
                c,
                r,
                str(adj_display[r, c]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    matrix_path = os.path.join(graph_dir, "adjacency_matrix.png")
    plt.savefig(matrix_path, dpi=300)
    plt.close()

    print(f"Saved {matrix_path}")