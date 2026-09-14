from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

USERS_PER_COMMUNITY = 10

COMMUNITIES = [
    {
        "name": "community_2",
        "probability_path": Path(
            r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_2_users_probabilities.csv"
        ),
        "request_path": Path(
            r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_2.csv"
        ),
    },
    {
        "name": "community_3",
        "probability_path": Path(
            r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_3_users_probabilities.csv"
        ),
        "request_path": Path(
            r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_3.csv"
        ),
    },
    # {
    #             "name": "community_0",
    #             "probability_path": Path(
    #                 r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_0_users_probabilities.csv"
    #             ),
    #             "request_path": Path(
    #                 r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_0.csv"
    #             ),
    #         },
    #     {
    #         "name": "community_1",
    #         "probability_path": Path(
    #             r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_1_users_probabilities.csv"
    #         ),
    #         "request_path": Path(
    #             r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_1.csv"
    #         ),
    #     },
    #     {
    #             "name": "community_4",
    #             "probability_path": Path(
    #                 r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_4_users_probabilities.csv"
    #             ),
    #             "request_path": Path(
    #                 r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_4.csv"
    #             ),
    #         },
    #         {
    #             "name": "community_5",
    #             "probability_path": Path(
    #                 r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_5_users_probabilities.csv"
    #             ),
    #             "request_path": Path(
    #                 r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_5.csv"
    #             ),
    #         },
    #         {
    #                 "name": "community_6",
    #                 "probability_path": Path(
    #                     r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_6_users_probabilities.csv"
    #                 ),
    #                 "request_path": Path(
    #                     r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_6.csv"
    #                 ),
    #             },
    #             {
    #                 "name": "community_7",
    #                 "probability_path": Path(
    #                     r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_7_users_probabilities.csv"
    #                 ),
    #                 "request_path": Path(
    #                     r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_7.csv"
    #                 ),
    #             },
        {
                "name": "community_8",
                "probability_path": Path(
                    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_8_users_probabilities.csv"
                ),
                "request_path": Path(
                    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_8.csv"
                ),
            },
        #     {
        #         "name": "community_9",
        #         "probability_path": Path(
        #             r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\louvain_communities\communities_csv\community_9_users_probabilities.csv"
        #         ),
        #         "request_path": Path(
        #             r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\request\requests_com_9.csv"
        #         ),
        #     },


]

OUTPUT_DIRECTORY = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\com\experiment"
)

REQUEST_FIRST_COLUMN_MODE = "auto"


# ============================================================
# FUNCTIONS
# ============================================================

def normalize_probabilities(probabilities, source_name):
    """Normalize every user probability vector to sum to 1."""

    row_sums = probabilities.sum(axis=1)

    if (row_sums <= 0).any():
        raise ValueError(
            f"{source_name} contains a zero-sum probability row."
        )

    return probabilities.div(row_sums, axis=0)


def load_probability_file(csv_path):
    """
    Load ALL users from a community probability CSV.

    Returns:
        full dataframe
        user ID column
        probability columns
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Probability file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(
            f"{csv_path.name} must contain a user column "
            "and probability columns."
        )

    user_id_column = df.columns[0]
    probability_columns = list(df.columns[1:])

    df[probability_columns] = df[
        probability_columns
    ].apply(pd.to_numeric, errors="raise")

    # Normalize every user's probability vector
    df[probability_columns] = normalize_probabilities(
        df[probability_columns],
        csv_path.name,
    )

    return (
        df,
        user_id_column,
        probability_columns,
    )


def normalise_column_name(value):

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
    )


def first_column_is_run_id(request_data):

    if request_data.shape[1] == 0:
        return False

    first_name = normalise_column_name(
        request_data.columns[0]
    )

    known_names = {
        "run",
        "runid",
        "run_id",
        "request",
        "requestid",
        "request_id",
        "index",
        "id",
        "unnamed:0",
        "unnamed:_0",
    }

    if (
        first_name in known_names
        or first_name.startswith("unnamed:")
    ):
        return True

    values = pd.to_numeric(
        request_data.iloc[:, 0],
        errors="coerce",
    )

    if values.isna().any():
        return False

    values = values.to_numpy(dtype=float)

    if not np.all(
        np.equal(values, np.floor(values))
    ):
        return False

    values = values.astype(np.int64)

    n = len(values)

    return (
        np.array_equal(
            values,
            np.arange(n),
        )
        or
        np.array_equal(
            values,
            np.arange(1, n + 1),
        )
    )


def find_request_column_for_user(
    columns,
    original_user_id,
):

    user_text = str(original_user_id).strip()

    try:
        numeric = float(user_text)

        if numeric.is_integer():
            user_text = str(int(numeric))

    except ValueError:
        pass

    candidates = {
        normalise_column_name(user_text),
        normalise_column_name(
            f"user_{user_text}"
        ),
        normalise_column_name(
            f"user{user_text}"
        ),
        normalise_column_name(
            f"u_{user_text}"
        ),
        normalise_column_name(
            f"u{user_text}"
        ),
    }

    for column in columns:

        if normalise_column_name(column) in candidates:
            return column

    return None


def load_selected_requests(
    csv_path,
    selected_original_user_ids,
    number_of_files,
):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Request file not found: {csv_path}"
        )

    request_data = pd.read_csv(csv_path)

    if request_data.empty:
        raise ValueError(
            f"Request file is empty: {csv_path}"
        )

    mode = REQUEST_FIRST_COLUMN_MODE.lower()

    has_run_id = (
        mode == "yes"
        or (
            mode == "auto"
            and first_column_is_run_id(
                request_data
            )
        )
    )

    # --------------------------------------------------------
    # RUN ID
    # --------------------------------------------------------

    if has_run_id:

        run_ids = pd.to_numeric(
            request_data.iloc[:, 0],
            errors="raise",
        ).astype(np.int64)

        request_data = (
            request_data
            .iloc[:, 1:]
            .copy()
        )

    else:

        run_ids = pd.Series(
            np.arange(
                1,
                len(request_data) + 1
            )
        )

    # --------------------------------------------------------
    # FIND SELECTED USERS IN REQUEST CSV
    # --------------------------------------------------------

    matched_columns = []

    for user_id in selected_original_user_ids:

        column = find_request_column_for_user(
            request_data.columns,
            user_id,
        )

        if column is None:
            matched_columns = []
            break

        matched_columns.append(column)

    if (
        matched_columns
        and len(set(matched_columns))
        == len(selected_original_user_ids)
    ):

        selected_requests = (
            request_data[
                matched_columns
            ].copy()
        )

    else:

        # Fallback:
        # use first USERS_PER_COMMUNITY request columns

        selected_requests = (
            request_data
            .iloc[
                :,
                :len(selected_original_user_ids)
            ]
            .copy()
        )

    selected_requests = (
        selected_requests
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .astype(np.int64)
    )

    minimum = int(
        selected_requests.min().min()
    )

    maximum = int(
        selected_requests.max().max()
    )

    if minimum < 0 or maximum >= number_of_files:

        raise ValueError(
            f"{csv_path.name}: request IDs range "
            f"from {minimum} to {maximum}, "
            f"but valid IDs are 0 to "
            f"{number_of_files - 1}."
        )

    return (
        selected_requests.reset_index(drop=True),
        run_ids.reset_index(drop=True),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(COMMUNITIES) < 2:
        raise ValueError(
            "At least two communities are required."
        )

    # Selected users only
    selected_probability_blocks = []

    # All users
    all_users_probability_blocks = []

    selected_request_blocks = []

    community_mean_vectors = []

    selected_user_mapping = []

    reference_probability_columns = None
    reference_run_ids = None

    next_new_user_id = 0

    # ========================================================
    # PROCESS EACH COMMUNITY
    # ========================================================

    for community_position, community in enumerate(
        COMMUNITIES
    ):

        community_name = community["name"]

        probability_path = (
            community["probability_path"]
        )

        request_path = (
            community["request_path"]
        )

        # ----------------------------------------------------
        # READ ALL USERS OF THIS COMMUNITY
        # ----------------------------------------------------

        (
            full_community_df,
            user_id_column,
            probability_columns,
        ) = load_probability_file(
            probability_path
        )

        if reference_probability_columns is None:

            reference_probability_columns = (
                probability_columns
            )

        elif (
            probability_columns
            != reference_probability_columns
        ):

            raise ValueError(
                f"{probability_path.name} does not "
                "have the same file columns."
            )

        number_of_files = len(
            probability_columns
        )

        total_users_in_community = len(
            full_community_df
        )

        if (
            total_users_in_community
            < USERS_PER_COMMUNITY
        ):

            raise ValueError(
                f"{community_name} contains only "
                f"{total_users_in_community} users, "
                f"but {USERS_PER_COMMUNITY} are required."
            )

        # ====================================================
        # ALL USERS
        #
        # Used ONLY for:
        #   community popularity
        #   global popularity
        # ====================================================

        all_users_probabilities = (
            full_community_df[
                probability_columns
            ]
            .copy()
            .reset_index(drop=True)
        )

        all_users_probability_blocks.append(
            all_users_probabilities
        )

        # ====================================================
        # COMMUNITY POPULARITY
        #
        # IMPORTANT:
        # Mean calculated using ALL USERS in this community
        # ====================================================

        community_mean = (
            all_users_probabilities.mean(axis=0)
        )

        community_mean = (
            community_mean
            / community_mean.sum()
        )

        community_mean_vectors.append(
            community_mean
        )

        # ====================================================
        # SELECT USERS FOR EXPERIMENT
        # ====================================================

        selected_df = (
            full_community_df
            .iloc[:USERS_PER_COMMUNITY]
            .copy()
        )

        selected_probabilities = (
            selected_df[
                probability_columns
            ]
            .copy()
            .reset_index(drop=True)
        )

        selected_probability_blocks.append(
            selected_probabilities
        )

        original_user_ids = (
            selected_df[
                user_id_column
            ].tolist()
        )

        # ====================================================
        # REQUESTS ONLY FOR SELECTED USERS
        # ====================================================

        (
            selected_requests,
            run_ids,
        ) = load_selected_requests(
            request_path,
            original_user_ids,
            number_of_files,
        )

        if reference_run_ids is None:

            reference_run_ids = run_ids

        elif not run_ids.equals(
            reference_run_ids
        ):

            raise ValueError(
                f"Run IDs in {request_path.name} "
                "do not match the other communities."
            )

        selected_request_blocks.append(
            selected_requests
        )

        # ====================================================
        # USER MAPPING
        # ====================================================

        for local_index, original_user_id in enumerate(
            original_user_ids
        ):

            selected_user_mapping.append(
                {
                    "new_user_id":
                        next_new_user_id,

                    "community":
                        community_name,

                    "community_position":
                        community_position,

                    "local_user_index":
                        local_index,

                    "original_user_id":
                        original_user_id,
                }
            )

            next_new_user_id += 1

        print(
            f"{community_name}: "
            f"{total_users_in_community} users used "
            "for popularity calculation; "
            f"{USERS_PER_COMMUNITY} selected "
            "for simulation."
        )

    # ========================================================
    # SELECTED USERS' INDIVIDUAL PREFERENCES
    # ========================================================

    user_preferences = pd.concat(
        selected_probability_blocks,
        ignore_index=True,
    )

    total_selected_users = len(
        user_preferences
    )

    new_user_ids = np.arange(
        total_selected_users
    )

    user_preferences_output = (
        user_preferences.copy()
    )

    user_preferences_output.insert(
        0,
        "user_id",
        new_user_ids,
    )

    # ========================================================
    # GLOBAL POPULARITY
    #
    # IMPORTANT:
    # calculated using ALL users from ALL selected communities
    # ========================================================

    all_users_all_communities = pd.concat(
        all_users_probability_blocks,
        ignore_index=True,
    )

    overall_global_vector = (
        all_users_all_communities.mean(axis=0)
    )

    overall_global_vector = (
        overall_global_vector
        / overall_global_vector.sum()
    )

    global_popularity_output = pd.DataFrame(
        np.tile(
            overall_global_vector.to_numpy(),
            (total_selected_users, 1),
        ),
        columns=reference_probability_columns,
    )

    global_popularity_output.insert(
        0,
        "user_id",
        new_user_ids,
    )

    # ========================================================
    # COMMUNITY POPULARITY
    #
    # Each selected user receives the mean vector calculated
    # from ALL users of their community.
    # ========================================================

    community_probability_blocks = []

    for community_mean in community_mean_vectors:

        block = pd.DataFrame(
            np.tile(
                community_mean.to_numpy(),
                (USERS_PER_COMMUNITY, 1),
            ),
            columns=reference_probability_columns,
        )

        community_probability_blocks.append(
            block
        )

    community_popularity_output = pd.concat(
        community_probability_blocks,
        ignore_index=True,
    )

    community_popularity_output.insert(
        0,
        "user_id",
        new_user_ids,
    )

    # ========================================================
    # COMBINED REQUESTS
    # ========================================================

    combined_requests = pd.concat(
        selected_request_blocks,
        axis=1,
        ignore_index=True,
    )

    combined_requests.columns = [
        f"user_{i}"
        for i in range(total_selected_users)
    ]

    combined_requests.insert(
        0,
        "run_id",
        reference_run_ids.to_numpy(),
    )

    # ========================================================
    # MAPPING
    # ========================================================

    mapping_output = pd.DataFrame(
        selected_user_mapping
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    user_preferences_path = (
        OUTPUT_DIRECTORY
        / "user_preferences.csv"
    )

    global_path = (
        OUTPUT_DIRECTORY
        / "global_popularity_all_communities.csv"
    )

    community_path = (
        OUTPUT_DIRECTORY
        / "global_popularity_per_community.csv"
    )

    requests_path = (
        OUTPUT_DIRECTORY
        / "combined_requests.csv"
    )

    mapping_path = (
        OUTPUT_DIRECTORY
        / "selected_users_mapping.csv"
    )

    user_preferences_output.to_csv(
        user_preferences_path,
        index=False,
        float_format="%.12f",
    )

    global_popularity_output.to_csv(
        global_path,
        index=False,
        float_format="%.12f",
    )

    community_popularity_output.to_csv(
        community_path,
        index=False,
        float_format="%.12f",
    )

    combined_requests.to_csv(
        requests_path,
        index=False,
    )

    mapping_output.to_csv(
        mapping_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nDone.")

    print(
        "\nTotal users used for GLOBAL calculation:",
        len(all_users_all_communities),
    )

    print(
        "Users selected for simulation:",
        total_selected_users,
    )

    print(
        "Number of files:",
        len(reference_probability_columns),
    )

    print(
        "Number of request runs:",
        len(reference_run_ids),
    )

    print("\nCreated:")

    print(user_preferences_path)
    print(global_path)
    print(community_path)
    print(requests_path)
    print(mapping_path)


if __name__ == "__main__":
    main()