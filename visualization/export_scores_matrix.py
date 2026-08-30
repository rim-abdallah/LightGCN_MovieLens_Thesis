import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_unique_path(path):
    if not os.path.exists(path):
        return path

    name, ext = os.path.splitext(path)
    counter = 1

    while True:
        new_path = f"{name}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def softmax_matrix(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def export_scores_matrix(
    sess,
    model,
    output_dir="output_scores",
    dataset_name="dataset",
):
    dataset_name = os.path.basename(dataset_name)

    excel_dir = os.path.join(output_dir, dataset_name, "excel")
    graph_dir = os.path.join(output_dir, dataset_name, "graphs")
    prob_dir = os.path.join(output_dir, dataset_name, "probabilities")

    shared_base_dir = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared"
    shared_dir = os.path.join(shared_base_dir, dataset_name)

    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(shared_dir, exist_ok=True)

    user_embs, item_embs = sess.run([
        model.ua_embeddings,
        model.ia_embeddings
    ])

    scores = np.dot(user_embs, item_embs.T)

    n_users = user_embs.shape[0]
    n_items = item_embs.shape[0]
    all_users = list(range(n_users))

    matrix = pd.DataFrame(
        scores,
        index=[f"User {u}" for u in all_users],
        columns=[f"Item {i}" for i in range(n_items)]
    )

    matrix_path = get_unique_path(os.path.join(
        excel_dir,
        "user_item_score_matrix.xlsx"
    ))

    matrix.to_excel(matrix_path)
    print(f"Full score matrix Excel saved to: {matrix_path}")

    prob_matrix = softmax_matrix(scores)

    prob_df = pd.DataFrame(
        prob_matrix,
        index=[f"User {u}" for u in all_users],
        columns=[f"Item {i}" for i in range(n_items)]
    )

    prob_path = get_unique_path(os.path.join(
        prob_dir,
        "user_item_probabilities.csv"
    ))

    prob_df.to_csv(prob_path)
    print(f"Probability CSV saved to: {prob_path}")

    shared_prob_path = get_unique_path(os.path.join(
        shared_dir,
        "user_item_probabilities.csv"
    ))

    prob_df.to_csv(shared_prob_path)
    print(f"Probability CSV also saved to Shared: {shared_prob_path}")

    print("First 5 probability row sums:")
    print(prob_df.sum(axis=1).head())

    '''graph_users = all_users[:10]

    all_rows = []

    for user_id in graph_users:
        user_scores = scores[user_id]
        sorted_items = np.argsort(user_scores)[::-1]

        for rank, item_id in enumerate(sorted_items, start=1):
            all_rows.append({
                "User_ID": user_id,
                "Rank": rank,
                "Item_ID": item_id,
                "Prediction_Score": float(user_scores[item_id])
            })

    df = pd.DataFrame(all_rows)

    fig, axes = plt.subplots(
        len(graph_users),
        1,
        figsize=(18, 4 * len(graph_users)),
        sharex=False
    )

    if len(graph_users) == 1:
        axes = [axes]

    for ax, user_id in zip(axes, graph_users):
        user_df = df[df["User_ID"] == user_id]

        item_ids = user_df["Item_ID"].values
        y_scores = user_df["Prediction_Score"].values
        x_positions = np.arange(len(item_ids))

        ax.scatter(x_positions, y_scores, s=5)
        ax.plot(x_positions, y_scores, linewidth=0.6)

        ax.set_title(f"User {user_id} - All Items Sorted by Prediction Score")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

        step = max(1, len(item_ids) // 20)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels(item_ids[::step], rotation=90)

    axes[-1].set_xlabel("Item ID")

    plt.suptitle("User-Item Prediction Scores for First 10 Users", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    scatter_path = get_unique_path(os.path.join(
        graph_dir,
        "user_item_connected_scatter_all_items.png"
    ))

    plt.savefig(scatter_path, dpi=300)
    plt.close()

    print(f"Graph saved to: {scatter_path}")'''