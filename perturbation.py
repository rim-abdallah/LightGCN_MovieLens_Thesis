import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# Change only these paths if needed
# =========================================================
INPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\user_item_probabilities.csv"

OUTPUT_FOLDER = (
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\with_error"
)

SEED = 42


def perturb_row(probabilities, epsilon, rng):
    """
    Changes one user's request-probability distribution.

    A total probability mass equal to epsilon is removed from some files
    and added to other files. Therefore, the Total Variation distance is
    exactly epsilon.
    """
    original = np.asarray(probabilities, dtype=float)
    perturbed = original.copy()

    if not np.isclose(original.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"One probability row does not sum to 1. Its sum is {original.sum()}."
        )

    # -----------------------------------------------------
    # Remove epsilon from randomly selected files
    # -----------------------------------------------------
    amount_left_to_remove = epsilon
    donor_indices = rng.permutation(np.where(perturbed > 0)[0])
    used_donors = set()

    for index in donor_indices:
        if amount_left_to_remove <= 1e-14:
            break

        removed_amount = min(perturbed[index], amount_left_to_remove)

        perturbed[index] -= removed_amount
        amount_left_to_remove -= removed_amount
        used_donors.add(index)

    if amount_left_to_remove > 1e-12:
        raise ValueError("Could not remove enough probability mass.")

    # -----------------------------------------------------
    # Add epsilon to different randomly selected files
    # This can turn a zero probability into a non-zero one.
    # -----------------------------------------------------
    amount_left_to_add = epsilon

    receiver_indices = np.array(
        [index for index in range(len(perturbed)) if index not in used_donors]
    )
    receiver_indices = rng.permutation(receiver_indices)

    for index in receiver_indices:
        if amount_left_to_add <= 1e-14:
            break

        available_capacity = 1.0 - perturbed[index]
        added_amount = min(available_capacity, amount_left_to_add)

        perturbed[index] += added_amount
        amount_left_to_add -= added_amount

    if amount_left_to_add > 1e-12:
        raise ValueError("Could not add enough probability mass.")

    # Correct only very small numerical rounding.
    perturbed = perturbed / perturbed.sum()

    # Verify the Total Variation distance.
    total_variation_distance = 0.5 * np.abs(original - perturbed).sum()

    if not np.isclose(total_variation_distance, epsilon, atol=1e-10):
        raise ValueError(
            f"TV distance is {total_variation_distance}, "
            f"but it should be {epsilon}."
        )

    return perturbed


def create_perturbed_csv(input_csv, output_csv, epsilon, seed):
    """
    Reads the original request-probability CSV and saves a perturbed version.
    """
    data = pd.read_csv(input_csv)

    # The first column is preserved if it contains user IDs such as "User 0".
    first_column_name = data.columns[0]

    if pd.api.types.is_numeric_dtype(data[first_column_name]):
        user_ids = None
        probability_data = data.copy()
    else:
        user_ids = data[[first_column_name]].copy()
        probability_data = data.iloc[:, 1:].copy()

    probability_data = probability_data.apply(pd.to_numeric, errors="raise")

    rng = np.random.default_rng(seed)

    perturbed_probabilities = []

    for _, row in probability_data.iterrows():
        perturbed_row = perturb_row(
            row.to_numpy(),
            epsilon,
            rng,
        )
        perturbed_probabilities.append(perturbed_row)

    result_probabilities = pd.DataFrame(
        perturbed_probabilities,
        columns=probability_data.columns,
    )

    if user_ids is not None:
        result = pd.concat(
            [user_ids.reset_index(drop=True), result_probabilities],
            axis=1,
        )
    else:
        result = result_probabilities

    result.to_csv(output_csv, index=False)

    print(f"Created: {output_csv}")
    print(f"Total Variation distance per user: {epsilon}")


# =========================================================
# Create the perturbed probability files
# =========================================================
output_folder = Path(OUTPUT_FOLDER)
output_folder.mkdir(parents=True, exist_ok=True)

create_perturbed_csv(
    input_csv=INPUT_CSV,
    output_csv=output_folder / "request_probabilities_tv_5_percent.csv",
    epsilon=0.05,
    seed=SEED,
)

create_perturbed_csv(
    input_csv=INPUT_CSV,
    output_csv=output_folder / "request_probabilities_tv_15_percent.csv",
    epsilon=0.15,
    seed=SEED + 1,
)