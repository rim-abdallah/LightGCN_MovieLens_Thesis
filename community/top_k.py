import os
import pandas as pd
import numpy as np


# ============================================================
# INPUT PERSONALIZED PROBABILITY CSV
# ============================================================

INPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_2_users_probabilities.csv"


# ============================================================
# OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\top_k"


# ============================================================
# TOP-K VALUES
# ============================================================

K_VALUES = [
    75,
    100,
    150,
    200,
    300,
    500,
    800,
    1000
]


def create_top_k(input_csv, output_folder, k_values):

    # Read personalized probabilities
    df = pd.read_csv(input_csv, index_col=0)

    print("Users:", df.shape[0])
    print("Files:", df.shape[1])

    os.makedirs(output_folder, exist_ok=True)

    # ========================================================
    # CREATE ONE CSV FOR EACH K
    # ========================================================

    for k in k_values:

        if k > df.shape[1]:
            print(f"Skipping K={k}: only {df.shape[1]} files available")
            continue

        # Start with zeros
        df_topk = pd.DataFrame(
            0.0,
            index=df.index,
            columns=df.columns
        )

        # ====================================================
        # FOR EACH USER
        # ====================================================

        for user in df.index:

            probabilities = df.loc[user]

            # Get this user's K highest-probability files
            top_items = probabilities.nlargest(k).index

            # Keep only these probabilities
            df_topk.loc[user, top_items] = probabilities[top_items]

        # ====================================================
        # NORMALIZE EACH USER
        # ====================================================

        row_sums = df_topk.sum(axis=1)

        df_topk = df_topk.div(
            row_sums.replace(0, np.nan),
            axis=0
        ).fillna(0)

        # ====================================================
        # SAVE
        # ====================================================

        output_csv = os.path.join(
            output_folder,
            f"personalized_top_{k}.csv"
        )

        df_topk.to_csv(output_csv)

        # Check
        print()
        print("==============================")
        print(f"K = {k}")
        print("==============================")
        print("Saved:", output_csv)
        print(
            "Average non-zero files/user:",
            (df_topk > 0).sum(axis=1).mean()
        )
        print(
            "Row sum min:",
            df_topk.sum(axis=1).min()
        )
        print(
            "Row sum max:",
            df_topk.sum(axis=1).max()
        )


create_top_k(
    INPUT_CSV,
    OUTPUT_FOLDER,
    K_VALUES
)