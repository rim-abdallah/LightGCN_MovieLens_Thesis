import os
import pandas as pd
import numpy as np


# ============================================================
# ROOT FOLDER
# ============================================================

ROOT = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\fixed com"


# ============================================================
# ALPHA VALUES
# ============================================================

ALPHAS = [0.25, 0.50, 0.75]


# ============================================================
# FILE NAMES INSIDE EACH "1st" FOLDER
# ============================================================

PREFERENCE_FILE = "user_preferences.csv"

GLOBAL_FILE = "global_popularity_all_communities.csv"

COMMUNITY_FILE = "global_popularity_per_community.csv"


# ============================================================
# FUNCTION TO MIX TWO PROBABILITY CSVs
# ============================================================

def create_alpha_probability(shared_path, preference_path, alpha, output_path):

    shared_df = pd.read_csv(shared_path)
    pref_df = pd.read_csv(preference_path)

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if shared_df.shape != pref_df.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"Shared: {shared_df.shape}\n"
            f"Preference: {pref_df.shape}"
        )

    if list(shared_df.columns) != list(pref_df.columns):
        raise ValueError(
            f"Column mismatch between:\n"
            f"{shared_path}\n"
            f"and\n"
            f"{preference_path}"
        )

    result_df = shared_df.copy()

    # --------------------------------------------------------
    # Detect probability columns
    #
    # This ignores columns such as user_id if they exist.
    # Only numeric columns are mixed.
    # --------------------------------------------------------

    numeric_cols = shared_df.select_dtypes(include=[np.number]).columns.tolist()

    # If there is a numeric user_id column, we do NOT want
    # to mix it as a probability.
    possible_id_columns = [
        "user_id",
        "user",
        "userid",
        "UserID",
        "User_ID"
    ]

    for col in possible_id_columns:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # --------------------------------------------------------
    # Alpha split
    #
    # alpha * shared probability
    # +
    # (1-alpha) * personalized probability
    # --------------------------------------------------------

    result_df[numeric_cols] = (
        alpha * shared_df[numeric_cols]
        +
        (1.0 - alpha) * pref_df[numeric_cols]
    )

    # --------------------------------------------------------
    # Normalize each row
    #
    # Normally this should already sum to 1 because it is
    # a convex combination of two probability distributions.
    # This is just for numerical safety.
    # --------------------------------------------------------

    row_sums = result_df[numeric_cols].sum(axis=1)

    result_df[numeric_cols] = (
        result_df[numeric_cols]
        .div(row_sums, axis=0)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result_df.to_csv(output_path, index=False)

    print(f"Created: {output_path}")

    # Check probability sums
    new_sums = result_df[numeric_cols].sum(axis=1)

    print(
        f"    alpha = {alpha}"
        f" | min row sum = {new_sums.min():.10f}"
        f" | max row sum = {new_sums.max():.10f}"
    )


# ============================================================
# PROCESS ALL COMMUNITY FOLDERS
# ============================================================

for folder_name in os.listdir(ROOT):

    community_folder = os.path.join(ROOT, folder_name)

    if not os.path.isdir(community_folder):
        continue

    # Example:
    # fixed com\2com\1st
    first_folder = os.path.join(community_folder, "1st")

    if not os.path.isdir(first_folder):
        print(f"Skipping {folder_name}: no '1st' folder")
        continue

    print("\n" + "=" * 70)
    print(f"Processing: {folder_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Input paths
    # --------------------------------------------------------

    preference_path = os.path.join(
        first_folder,
        PREFERENCE_FILE
    )

    global_path = os.path.join(
        first_folder,
        GLOBAL_FILE
    )

    community_path = os.path.join(
        first_folder,
        COMMUNITY_FILE
    )

    # Make sure all required files exist
    required_files = [
        preference_path,
        global_path,
        community_path
    ]

    missing = [path for path in required_files if not os.path.exists(path)]

    if missing:
        print("Missing files:")
        for path in missing:
            print("   ", path)

        print(f"Skipping {folder_name}")
        continue

    # ========================================================
    # CREATE OUTPUT FOLDERS
    # ========================================================

    alpha_folder = os.path.join(
        first_folder,
        "alpha"
    )

    com_folder = os.path.join(
        alpha_folder,
        "com"
    )

    glob_folder = os.path.join(
        alpha_folder,
        "glob"
    )

    os.makedirs(com_folder, exist_ok=True)
    os.makedirs(glob_folder, exist_ok=True)

    # ========================================================
    # CREATE ALPHA FILES
    # ========================================================

    for alpha in ALPHAS:

        alpha_name = str(alpha).rstrip("0").rstrip(".")

        # ----------------------------------------------------
        # GLOBAL + PREFERENCE
        # ----------------------------------------------------

        global_output = os.path.join(
            glob_folder,
            f"cache_alpha_{alpha_name}.csv"
        )

        create_alpha_probability(
            shared_path=global_path,
            preference_path=preference_path,
            alpha=alpha,
            output_path=global_output
        )

        # ----------------------------------------------------
        # COMMUNITY + PREFERENCE
        # ----------------------------------------------------

        community_output = os.path.join(
            com_folder,
            f"cache_alpha_{alpha_name}.csv"
        )

        create_alpha_probability(
            shared_path=community_path,
            preference_path=preference_path,
            alpha=alpha,
            output_path=community_output
        )


print("\nDone.")