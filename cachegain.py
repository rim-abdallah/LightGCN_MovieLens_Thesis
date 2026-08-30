import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# INPUTS
# ============================================================

# Request probabilities p[u,f]
PUF_PATH = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\just_theory\all_users_pref_probabilities.csv"
)

# Cache-placement probabilities q[u,f]
QUF_PATH = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\just_theory\all_users_community_probabilities.csv"
)


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

U = 608
N = 1000
B = 100

# Same CACHE_SIZE used in C++
CACHE_SIZE = 5

# Same M_max used in the simulation
M_MAX = 70

REMOVE_FIRST_COLUMN_PUF = True
REMOVE_FIRST_COLUMN_QUF = True


# ============================================================
# READ CSV FILES
# ============================================================

puf_df = pd.read_csv(PUF_PATH)
quf_df = pd.read_csv(QUF_PATH)

if REMOVE_FIRST_COLUMN_PUF:
    puf_df = puf_df.iloc[:, 1:]

if REMOVE_FIRST_COLUMN_QUF:
    quf_df = quf_df.iloc[:, 1:]

puf = puf_df.to_numpy(dtype=float)
quf = quf_df.to_numpy(dtype=float)


# ============================================================
# TAKE EXACTLY U USERS AND N FILES
# ============================================================

if puf.shape[0] < U or puf.shape[1] < N:
    raise ValueError(
        f"PUF shape = {puf.shape}, expected at least ({U}, {N})"
    )

if quf.shape[0] < U or quf.shape[1] < N:
    raise ValueError(
        f"QUF shape = {quf.shape}, expected at least ({U}, {N})"
    )

puf = puf[:U, :N]
quf = quf[:U, :N]


# ============================================================
# CHECK PROBABILITIES
# ============================================================

p_sums = np.sum(puf, axis=1)
q_sums = np.sum(quf, axis=1)

print("=" * 70)
print("PROBABILITY CHECK")
print("=" * 70)

print(
    f"p[u,f] row sum: "
    f"mean={np.mean(p_sums):.12f}, "
    f"min={np.min(p_sums):.12f}, "
    f"max={np.max(p_sums):.12f}"
)

print(
    f"q[u,f] row sum: "
    f"mean={np.mean(q_sums):.12f}, "
    f"min={np.min(q_sums):.12f}, "
    f"max={np.max(q_sums):.12f}"
)

if not np.allclose(p_sums, 1.0, atol=1e-6):
    raise ValueError("PUF rows do not sum to 1.")

if not np.allclose(q_sums, 1.0, atol=1e-6):
    raise ValueError("QUF rows do not sum to 1.")


# ============================================================
# CACHE SIZE
# ============================================================
#
# Same as C++:
#
# M_cache = CACHE_SIZE * m_files / 100
# B_cache = M_cache * b_chunks
#
# ============================================================

M_cache_files = (CACHE_SIZE * N) // 100
B_cache = M_cache_files * B

print()
print("=" * 70)
print("CACHE PARAMETERS")
print("=" * 70)

print(f"Users K                 = {U}")
print(f"Files N                 = {N}")
print(f"Chunks/file B           = {B}")
print(f"Cache percentage        = {CACHE_SIZE}%")
print(f"Cache size in files     = {M_cache_files}")
print(f"Cache size in chunks    = {B_cache}")
print(f"Maximum chunks/file     = {M_MAX}")


# ============================================================
# EXACT M[u,f] ALLOCATION USED IN C++
# ============================================================

def allocate_cache_cpp(q_user):

    m_user = np.zeros(N, dtype=int)

    chunk_check = 0
    remainders = []

    # --------------------------------------------------------
    # INITIAL FLOOR ALLOCATION
    # --------------------------------------------------------

    for f in range(N):

        if q_user[f] <= 0.0:
            m_user[f] = 0
            continue

        raw = B_cache * q_user[f]

        m_user[f] = int(np.floor(raw))

        if m_user[f] > M_MAX:
            m_user[f] = M_MAX

        chunk_check += m_user[f]

        decimal_part = raw - np.floor(raw)

        remainders.append(
            (
                decimal_part,
                f
            )
        )

    # Same as:
    #
    # sort(... greater<pair<double,unsigned int>>())
    #
    remainders.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )

    remaining = B_cache - chunk_check

    # --------------------------------------------------------
    # REDISTRIBUTE REMAINING CACHE
    # --------------------------------------------------------

    while remaining > 0:

        added = False

        for decimal_part, file_id in remainders:

            if remaining <= 0:
                break

            if (
                q_user[file_id] > 0.0
                and
                m_user[file_id] < M_MAX
            ):
                m_user[file_id] += 1
                remaining -= 1
                added = True

        if not added:

            print(
                "WARNING: cannot fill full cache. "
                f"Remaining chunks = {remaining}"
            )

            break

    return m_user


# ============================================================
# GENERATE M[u,f]
# ============================================================

muf = np.zeros(
    (U, N),
    dtype=int
)

for u in range(U):

    muf[u, :] = allocate_cache_cpp(
        quf[u, :]
    )


# ============================================================
# CACHE CHECK
# ============================================================

cache_used = np.sum(
    muf,
    axis=1
)

print()
print("=" * 70)
print("CACHE ALLOCATION CHECK")
print("=" * 70)

print(f"Target chunks/user      = {B_cache}")
print(f"Minimum chunks/user     = {np.min(cache_used)}")
print(f"Maximum chunks/user     = {np.max(cache_used)}")
print(f"Average chunks/user     = {np.mean(cache_used):.6f}")
print(f"Maximum m[u,f] observed = {np.max(muf)}")


# ============================================================
# EXPECTED T0
# ============================================================
#
# E[T0]
#
# = B sum_f [
#
#       1 - product_u(1-p[u,f])
#
#   ]
#
# ============================================================

product_no_request = np.prod(
    1.0 - puf,
    axis=0
)

prob_file_requested = (
    1.0
    - product_no_request
)

E_T0 = (
    B
    * np.sum(
        prob_file_requested
    )
)


# ============================================================
# METHOD 1
#
# OLD T_u:
# UNIFORM CHUNK-SELECTION ASSUMPTION
# ============================================================
#
# Assumption:
#
# P(chunk cached by user u)
#
#       = m[u,f] / B
#
#
# E[Tu]
#
# = B sum_f [
#
#       1 -
#       product_u(
#
#           1 -
#           p[u,f] *
#           (1-m[u,f]/B)
#
#       )
#
#   ]
#
# ============================================================

missing_fraction = (
    1.0
    -
    muf.astype(float) / B
)

prob_user_needs_chunk = (
    puf
    *
    missing_fraction
)

product_no_user_needs_chunk = np.prod(
    1.0
    -
    prob_user_needs_chunk,
    axis=0
)

prob_chunk_needed = (
    1.0
    -
    product_no_user_needs_chunk
)

E_Tu_uniform = (
    B
    *
    np.sum(
        prob_chunk_needed
    )
)

G_uniform = (
    1.0
    -
    E_Tu_uniform / E_T0
)


# ============================================================
# METHOD 2
#
# CURRENT C++ round() CHUNK-SELECTION MODEL
# ============================================================
#
# C++ currently does:
#
# idx_rand = round(
#     randomNumber(
#         0,
#         n_list - 1
#     )
# );
#
#
# This means the list positions do NOT have equal probability.
#
# If n chunks remain:
#
# First position:
#
#       0.5 / (n-1)
#
# Last position:
#
#       0.5 / (n-1)
#
# Interior position:
#
#       1 / (n-1)
#
#
# We calculate:
#
# pi[m,k]
#
# = probability chunk ID k is cached
#   after selecting m chunks using
#   the CURRENT C++ round() procedure.
#
# ============================================================

def build_cpp_round_probability(B, M_MAX):

    pi = np.zeros(
        (M_MAX + 1, B),
        dtype=float
    )

    # --------------------------------------------------------
    # Calculate probability separately for every chunk ID
    # --------------------------------------------------------

    for chunk_id in range(B):

        # ----------------------------------------------------
        # IMPORTANT
        #
        # C++ builds the linked list like:
        #
        # k = 0 -> [0]
        # k = 1 -> [1,0]
        # k = 2 -> [2,1,0]
        #
        # Therefore final initial order is:
        #
        # B-1, B-2, ..., 1, 0
        #
        # For chunk k:
        #
        # initial position = B-1-k
        #
        # ----------------------------------------------------

        number_before = (
            B - 1 - chunk_id
        )

        number_after = (
            chunk_id
        )

        # State:
        #
        # (a,b)
        #
        # a = remaining chunks before target
        # b = remaining chunks after target
        #
        # State probability means target chunk has NOT
        # yet been selected.
        #
        states = {
            (
                number_before,
                number_after
            ): 1.0
        }

        # ----------------------------------------------------
        # Perform m sequential selections
        # ----------------------------------------------------

        for m in range(
            1,
            M_MAX + 1
        ):

            new_states = {}

            for (
                a,
                b
            ), state_probability in states.items():

                n = (
                    a
                    +
                    b
                    +
                    1
                )

                # ------------------------------------------------
                # Only target remains
                # ------------------------------------------------

                if n == 1:

                    p_target = 1.0
                    p_before = 0.0
                    p_after = 0.0

                else:

                    denominator = (
                        n - 1
                    )

                    # --------------------------------------------
                    # TARGET SELECTION PROBABILITY
                    #
                    # If target is first or last:
                    #
                    # 0.5/(n-1)
                    #
                    # Otherwise:
                    #
                    # 1/(n-1)
                    # --------------------------------------------

                    if (
                        a == 0
                        or
                        b == 0
                    ):

                        p_target = (
                            0.5
                            /
                            denominator
                        )

                    else:

                        p_target = (
                            1.0
                            /
                            denominator
                        )

                    # --------------------------------------------
                    # Probability some position BEFORE target
                    # gets selected.
                    #
                    # The global first position has half weight.
                    # --------------------------------------------

                    if a > 0:

                        p_before = (
                            a - 0.5
                        ) / denominator

                    else:

                        p_before = 0.0

                    # --------------------------------------------
                    # Probability some position AFTER target
                    # gets selected.
                    #
                    # Global last position has half weight.
                    # --------------------------------------------

                    if b > 0:

                        p_after = (
                            b - 0.5
                        ) / denominator

                    else:

                        p_after = 0.0

                # ------------------------------------------------
                # If a chunk BEFORE target is selected,
                # target moves one position toward front.
                # ------------------------------------------------

                if (
                    a > 0
                    and
                    p_before > 0.0
                ):

                    next_state = (
                        a - 1,
                        b
                    )

                    new_states[next_state] = (
                        new_states.get(
                            next_state,
                            0.0
                        )
                        +
                        state_probability
                        *
                        p_before
                    )

                # ------------------------------------------------
                # If a chunk AFTER target is selected,
                # number before target stays unchanged.
                # ------------------------------------------------

                if (
                    b > 0
                    and
                    p_after > 0.0
                ):

                    next_state = (
                        a,
                        b - 1
                    )

                    new_states[next_state] = (
                        new_states.get(
                            next_state,
                            0.0
                        )
                        +
                        state_probability
                        *
                        p_after
                    )

                # ------------------------------------------------
                # If the target itself is selected,
                # we DO NOT add a new state.
                #
                # That probability corresponds to the target
                # being cached.
                # ------------------------------------------------

            states = new_states

            # Probability target was never selected
            probability_not_cached = sum(
                states.values()
            )

            # Probability target IS cached
            pi[
                m,
                chunk_id
            ] = (
                1.0
                -
                probability_not_cached
            )

    return pi


# ============================================================
# BUILD pi[m,k]
# ============================================================

cpp_round_pi = (
    build_cpp_round_probability(
        B,
        M_MAX
    )
)


# ============================================================
# CHECK pi[m,k]
# ============================================================
#
# Since exactly m chunks are selected:
#
# sum_k pi[m,k] should equal m.
#
# ============================================================

print()
print("=" * 70)
print("CURRENT C++ round() PROBABILITY CHECK")
print("=" * 70)

test_values = [
    1,
    min(10, M_MAX),
    M_MAX
]

for m_test in test_values:

    print()
    print(
        f"m = {m_test}"
    )

    print(
        f"Sum P(cached) = "
        f"{np.sum(cpp_round_pi[m_test]):.12f}"
    )

    print(
        f"Expected sum   = "
        f"{m_test}"
    )

    print(
        f"Average P      = "
        f"{np.mean(cpp_round_pi[m_test]):.12f}"
    )

    print(
        f"Uniform m/B    = "
        f"{m_test / B:.12f}"
    )

    print(
        f"Minimum P      = "
        f"{np.min(cpp_round_pi[m_test]):.12f}"
    )

    print(
        f"Maximum P      = "
        f"{np.max(cpp_round_pi[m_test]):.12f}"
    )


# ============================================================
# EXPECTED T_u USING CURRENT C++ round()
# ============================================================
#
# Now each chunk k has its own cache probability:
#
#       pi[m[u,f], k]
#
#
# User u needs chunk k of file f with probability:
#
#       p[u,f] *
#       (1 - pi[m[u,f],k])
#
#
# Probability NO user needs it:
#
#       product_u [
#
#           1 -
#           p[u,f] *
#           (1-pi[m[u,f],k])
#
#       ]
#
#
# Therefore:
#
# E[Tu]
#
# = sum_f sum_k [
#
#       1 -
#       product_u(
#
#           1 -
#           p[u,f] *
#           (1-pi[m[u,f],k])
#
#       )
#
#   ]
#
# ============================================================

E_Tu_cpp_round = 0.0


for f in range(N):

    # --------------------------------------------------------
    # Number of cached chunks for each user for this file
    #
    # Shape: U
    # --------------------------------------------------------

    m_for_users = (
        muf[:, f]
    )

    # --------------------------------------------------------
    # Probability each user caches each chunk
    #
    # Shape:
    #
    # U x B
    #
    # --------------------------------------------------------

    cache_probability = (
        cpp_round_pi[
            m_for_users,
            :
        ]
    )

    # --------------------------------------------------------
    # p[u,f]
    #
    # Convert to:
    #
    # U x 1
    #
    # --------------------------------------------------------

    request_probability = (
        puf[:, f]
        .reshape(
            U,
            1
        )
    )

    # --------------------------------------------------------
    # Probability user u needs each chunk
    # --------------------------------------------------------

    probability_user_needs = (
        request_probability
        *
        (
            1.0
            -
            cache_probability
        )
    )

    # --------------------------------------------------------
    # Probability nobody needs each chunk
    # --------------------------------------------------------

    probability_no_user_needs = np.prod(
        1.0
        -
        probability_user_needs,
        axis=0
    )

    # --------------------------------------------------------
    # Probability at least one user needs chunk k
    # --------------------------------------------------------

    probability_chunk_needed = (
        1.0
        -
        probability_no_user_needs
    )

    # --------------------------------------------------------
    # Sum over B chunks
    # --------------------------------------------------------

    E_Tu_cpp_round += np.sum(
        probability_chunk_needed
    )


# ============================================================
# GAIN USING CURRENT C++ round() T_u
# ============================================================

G_cpp_round = (
    1.0
    -
    E_Tu_cpp_round
    /
    E_T0
)


# ============================================================
# SIMPLE SUM APPROXIMATION
# ============================================================
#
# G_sum
#
# ~= 1/(BK)
#
#    sum_u sum_f
#
#    p[u,f] m[u,f]
#
# ============================================================

sum_p_m = np.sum(
    puf
    *
    muf
)

G_sum = (
    sum_p_m
    /
    (
        B * U
    )
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("EXPECTED LOCAL CACHE RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# T0
# ------------------------------------------------------------

print()
print("--- EXPECTED T0 ---")

print(
    f"E[T0] = "
    f"{E_T0:.12f}"
)


# ------------------------------------------------------------
# OLD UNIFORM MODEL
# ------------------------------------------------------------

print()
print("--- OLD UNIFORM CHUNK MODEL ---")

print(
    f"E[Tu] uniform = "
    f"{E_Tu_uniform:.12f}"
)

print(
    f"E[Tu] / E[T0] = "
    f"{E_Tu_uniform / E_T0:.12f}"
)

print(
    f"Cache gain = "
    f"{G_uniform:.12f}"
)

print(
    f"Cache gain (%) = "
    f"{100 * G_uniform:.6f}%"
)


# ------------------------------------------------------------
# CURRENT C++ round() MODEL
# ------------------------------------------------------------

print()
print("--- CURRENT C++ round() MODEL ---")

print(
    f"E[Tu] round = "
    f"{E_Tu_cpp_round:.12f}"
)

print(
    f"E[Tu] / E[T0] = "
    f"{E_Tu_cpp_round / E_T0:.12f}"
)

print(
    f"Cache gain = "
    f"{G_cpp_round:.12f}"
)

print(
    f"Cache gain (%) = "
    f"{100 * G_cpp_round:.6f}%"
)


# ------------------------------------------------------------
# SIMPLE SUM
# ------------------------------------------------------------

print()
print("--- SIMPLE SUM APPROXIMATION ---")

print(
    f"sum p[u,f] * m[u,f] = "
    f"{sum_p_m:.12f}"
)

print(
    f"Approximate cache gain = "
    f"{G_sum:.12f}"
)

print(
    f"Approximate cache gain (%) = "
    f"{100 * G_sum:.6f}%"
)


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    f"Uniform-model Tu        = "
    f"{E_Tu_uniform:.6f}"
)

print(
    f"C++ round-model Tu      = "
    f"{E_Tu_cpp_round:.6f}"
)

print(
    f"Difference in Tu        = "
    f"{E_Tu_cpp_round - E_Tu_uniform:.6f}"
)

print()

print(
    f"Uniform-model gain      = "
    f"{100 * G_uniform:.6f}%"
)

print(
    f"C++ round-model gain    = "
    f"{100 * G_cpp_round:.6f}%"
)

print(
    f"Sum approximation gain  = "
    f"{100 * G_sum:.6f}%"
)

print(
    f"Round-vs-uniform diff   = "
    f"{100 * (G_cpp_round - G_uniform):.6f} "
    "percentage points"
)

print("=" * 70)