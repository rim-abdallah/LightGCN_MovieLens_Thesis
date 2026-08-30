import os
import numpy as np
import matplotlib.pyplot as plt

from visualization.visualize_interest_graph import (
    visualize_interest_graph_by_distance
)


def show_user_top_items(
    sess,
    model,
    sampled_users,
    top_k=10,
    draw_graph=True,
    output_dir="output_scores"
):
    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    user_embs, item_embs = sess.run([
        model.ua_embeddings,
        model.ia_embeddings
    ])

    lines = []
    all_top_items = set()

    for user_id in sampled_users:
        user_vec = user_embs[user_id]

        scores = np.dot(item_embs, user_vec)

        top_items = np.argsort(scores)[::-1][:top_k]

        all_top_items.update(top_items)

        lines.append(f"User {user_id}")
        lines.append("")

        for rank, item_id in enumerate(top_items, start=1):
            lines.append(
                f"{rank}. Item {item_id} "
                f"-> score = {scores[item_id]:.4f}"
            )

        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    text_output = "\n".join(lines)

    plt.figure(figsize=(10, 10))
    plt.axis("off")

    plt.text(
        0.01,
        0.99,
        text_output,
        fontsize=11,
        family="monospace",
        verticalalignment="top"
    )

    plt.tight_layout()

    ranking_path = os.path.join(
        graph_dir,
        "ranking_list.png"
    )

    plt.savefig(
        ranking_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {ranking_path}")

    # Draw interest graph using all top-ranked items
    if draw_graph:
        visualize_interest_graph_by_distance(
            sess=sess,
            model=model,
            data_generator=None,
            sampled_users=sampled_users,
            sampled_items=list(all_top_items),
            graph_name="ranking_top_items",
            output_dir=output_dir
        )
        
        
        '''graph ranking list +interest graph ranking top item
        tenye heye graph ke birtite bass user m3 top item t3oulo '''