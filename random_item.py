import os
import numpy as np
import pandas as pd

# =====================================================
# INPUT / OUTPUT PATHS
# =====================================================
INPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens\user_item_probabilities_2.csv"

OUTPUT_FOLDER = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\Movie_Lens"

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "user_item_probabilities_1000.csv"
)

MAPPING_CSV = os.path.join(
    OUTPUT_FOLDER,
    "file_id_mapping.csv"
)

# =====================================================
# PARAMETERS
# =====================================================
N_ITEMS = 1000
RANDOM_STATE = 42

# =====================================================
# READ CSV
# =====================================================
df = pd.read_csv(INPUT_CSV, index_col=0)

print("Original shape:", df.shape)

n_users, n_files = df.shape

if N_ITEMS > n_files:
    raise ValueError(
        f"Dataset has only {n_files} items."
    )

# =====================================================
# RANDOMLY SELECT 1000 ITEMS
# =====================================================
selected_columns = np.random.RandomState(RANDOM_STATE).choice(
    df.columns,
    size=N_ITEMS,
    replace=False
)

df_selected = df[selected_columns].copy()

# =====================================================
# SAVE ORIGINAL -> NEW ID MAPPING
# =====================================================
mapping = pd.DataFrame({
    "New_File_ID": range(N_ITEMS),
    "Original_File_ID": selected_columns
})

mapping.to_csv(MAPPING_CSV, index=False)

# =====================================================
# RENAME ITEMS FROM 0 TO 999
# =====================================================
df_selected.columns = [str(i) for i in range(N_ITEMS)]

# =====================================================
# NORMALIZE EACH USER
# =====================================================
row_sum = df_selected.sum(axis=1)

df_selected = df_selected.div(
    row_sum.replace(0, np.nan),
    axis=0
).fillna(0)

# =====================================================
# VERIFY NORMALIZATION
# =====================================================
check = df_selected.sum(axis=1)

print("Minimum row sum:", check.min())
print("Maximum row sum:", check.max())

# =====================================================
# SAVE CSV
# =====================================================
df_selected.to_csv(OUTPUT_CSV)

print()
print("Saved normalized probability file:")
print(OUTPUT_CSV)

print()
print("Saved mapping file:")
print(MAPPING_CSV)

print()
print("Final shape:", df_selected.shape)