import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def dot_score(user_vec, item_vec):
    return float(np.dot(user_vec, item_vec))


def score_to_distance(score):
    # Higher positive score means shorter visual distance
    return float(1 / (1 + max(score, 0)))


def visualize_interest_graph_by_distance(
    sess,
    model,
    data_generator,
    sampled_users,
    sampled_items,
    graph_name="sample",
    output_dir="output_scores"
):
    print("Starting interest graph...")

    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    user_embs, item_embs = sess.run([
        model.ua_embeddings,
        model.ia_embeddings
    ])

    users = {f"U{u}": user_embs[u] for u in sampled_users}
    items = {f"I{i}": item_embs[i] for i in sampled_items}

    G = nx.Graph()

    user_colors = [
        "red", "blue", "green", "purple",
        "brown", "pink", "olive", "cyan"
    ]

    user_color_map = {
        u_name: user_colors[idx % len(user_colors)]
        for idx, u_name in enumerate(users.keys())
    }

    for u_name in users:
        G.add_node(
            u_name,
            color=user_color_map[u_name],
            node_type="User"
        )

    for i_name in items:
        G.add_node(
            i_name,
            color="orange",
            node_type="Item"
        )

    for u_name, u_vec in users.items():
        for i_name, i_vec in items.items():
            score = dot_score(u_vec, i_vec)
            distance = score_to_distance(score)
            edge_width = 1 + abs(score) * 0.15

            G.add_edge(
                u_name,
                i_name,
                score=score,
                distance=distance,
                edge_color=user_color_map[u_name],
                edge_width=edge_width
            )

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    pos = nx.spring_layout(
        G,
        weight="distance",
        k=1.0,
        iterations=80,
        seed=42
    )

    plt.figure(figsize=(12, 8))

    node_colors = [G.nodes[n]["color"] for n in G.nodes]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=900
    )

    edges = list(G.edges())

    edge_colors = [
        G.edges[u, v]["edge_color"]
        for u, v in edges
    ]

    edge_widths = [
        G.edges[u, v]["edge_width"]
        for u, v in edges
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=edge_widths,
        alpha=0.45,
        edge_color=edge_colors
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold"
    )

    edge_labels = {
        (u, v): (
            f"S:{G.edges[u, v]['score']:.2f}\n"
            f"D:{G.edges[u, v]['distance']:.2f}"
        )
        for u, v in edges
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5
    )

    plt.title(
        "Interest Graph Using Dot-Product Score\n"
        "Higher Score = Shorter Displayed Distance"
    )

    plt.axis("off")
    plt.tight_layout()

    filename = f"interest_graph_{graph_name}.png"
    save_path = os.path.join(graph_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Interest graph saved as {save_path}")