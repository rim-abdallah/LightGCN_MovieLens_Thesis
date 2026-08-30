from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# INPUT AND OUTPUT FOLDERS
# ============================================================

BASE_FOLDER = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens"
    r"\Shared\Movie_Lens\louvain_communities\communities_csv"
)

OUTPUT_FOLDER = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens"
    r"\Shared\Movie_Lens\community_similarity"
)

# Set to True if the first CSV column contains the user ID.
IGNORE_FIRST_COLUMN = True


# ============================================================
# FUNCTIONS
# ============================================================

def extract_community_number(csv_path: Path) -> int:
    """
    Extract the community number from a filename such as:
    community_5_users_probabilities.csv
    """
    match = re.search(
        r"community_(\d+)_users_probabilities\.csv$",
        csv_path.name,
        flags=re.IGNORECASE
    )

    if match is None:
        raise ValueError(
            f"Invalid community filename: {csv_path.name}"
        )

    return int(match.group(1))


def load_community_average(csv_path: Path) -> pd.Series:
    """
    Load one community CSV and calculate the average
    preference vector across all users in that community.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"The CSV file is empty: {csv_path}"
        )

    if IGNORE_FIRST_COLUMN:
        if df.shape[1] < 2:
            raise ValueError(
                f"The CSV does not contain probability columns: "
                f"{csv_path}"
            )

        df = df.iloc[:, 1:]

    # Convert probability columns to numeric values.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove columns that do not contain any numeric values.
    df = df.dropna(axis=1, how="all")

    if df.empty:
        raise ValueError(
            f"No valid probability columns were found in: "
            f"{csv_path}"
        )

    # Replace missing values by zero.
    df = df.fillna(0.0)

    # Return the mean file-preference vector of the community.
    return df.mean(axis=0)


def cosine_similarity(
    vector_a: pd.Series,
    vector_b: pd.Series
) -> float:
    """
    Calculate cosine similarity between two aligned vectors.
    """
    a = vector_a.to_numpy(dtype=float)
    b = vector_b.to_numpy(dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def column_sort_key(column_name):
    """
    Sort numeric file columns numerically and other
    columns alphabetically.
    """
    column_text = str(column_name)

    if column_text.isdigit():
        return 0, int(column_text)

    return 1, column_text


# ============================================================
# FIND COMMUNITY FILES
# ============================================================

if not BASE_FOLDER.exists():
    raise FileNotFoundError(
        f"Community folder not found:\n{BASE_FOLDER}"
    )

community_files = list(
    BASE_FOLDER.glob("community_*_users_probabilities.csv")
)

if not community_files:
    raise FileNotFoundError(
        "No community CSV files were found in:\n"
        f"{BASE_FOLDER}"
    )

# Sort files according to their community number.
community_files.sort(key=extract_community_number)

print(f"Found {len(community_files)} community files.\n")


# ============================================================
# LOAD COMMUNITY AVERAGES
# ============================================================

community_vectors = {}

for csv_path in community_files:
    community_number = extract_community_number(csv_path)
    community_name = f"Community_{community_number}"

    print(f"Processing {community_name}")
    print(f"File: {csv_path}")

    community_vectors[community_name] = (
        load_community_average(csv_path)
    )


# ============================================================
# ALIGN FILE COLUMNS
# ============================================================

common_columns = None

for vector in community_vectors.values():
    current_columns = set(vector.index)

    if common_columns is None:
        common_columns = current_columns
    else:
        common_columns = common_columns.intersection(
            current_columns
        )

if not common_columns:
    raise ValueError(
        "The community CSV files do not have any common "
        "probability columns."
    )

common_columns = sorted(
    common_columns,
    key=column_sort_key
)

for community_name in community_vectors:
    community_vectors[community_name] = (
        community_vectors[community_name]
        .reindex(common_columns)
        .fillna(0.0)
    )

print(
    f"\nNumber of common file columns: "
    f"{len(common_columns)}"
)


# ============================================================
# CALCULATE SIMILARITY MATRIX
# ============================================================

community_names = list(community_vectors.keys())

similarity_matrix = pd.DataFrame(
    index=community_names,
    columns=community_names,
    dtype=float
)

for community_a in community_names:
    for community_b in community_names:
        similarity_matrix.loc[
            community_a,
            community_b
        ] = cosine_similarity(
            community_vectors[community_a],
            community_vectors[community_b]
        )


# ============================================================
# CREATE UNIQUE COMMUNITY PAIRS
# ============================================================

pairwise_results = []

for i, community_a in enumerate(community_names):
    for j in range(i + 1, len(community_names)):
        community_b = community_names[j]

        similarity = similarity_matrix.loc[
            community_a,
            community_b
        ]

        pairwise_results.append({
            "Community_1": community_a,
            "Community_2": community_b,
            "Cosine_Similarity": similarity
        })

pairwise_df = pd.DataFrame(pairwise_results)

# Sort pairs from the most similar to the least similar.
pairwise_df = pairwise_df.sort_values(
    by="Cosine_Similarity",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

matrix_output = (
    OUTPUT_FOLDER / "community_similarity_matrix.csv"
)

pairs_output = (
    OUTPUT_FOLDER / "community_similarity_pairs.csv"
)

averages_output = (
    OUTPUT_FOLDER / "community_average_probabilities.csv"
)

similarity_matrix.to_csv(matrix_output)

pairwise_df.to_csv(
    pairs_output,
    index=False
)

average_probabilities = pd.DataFrame(
    community_vectors
).T

average_probabilities.index.name = "Community"

average_probabilities.to_csv(
    averages_output
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nCommunity cosine-similarity matrix:")
print(similarity_matrix.round(6))

print("\nCommunity pairs, from most similar to least similar:")
print(pairwise_df.round(6).to_string(index=False))

print("\nFiles saved successfully:")
print(matrix_output)
print(pairs_output)
print(averages_output)