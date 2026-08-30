import os
import pandas as pd

# ==========================================
# INPUT / OUTPUT
# ==========================================
INPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\zipf\Shared\Movie_Lens\zipf_communities\community_0_zipf.csv"

OUTPUT_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\zipf\Shared\Movie_Lens\zipf_communities\community_0_zipf_nonzero_renamed.csv"

MAPPING_CSV = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\zipf\Shared\Movie_Lens\zipf_communities\community_0_file_mapping.csv"

# ==========================================
# READ CSV
# ==========================================
df = pd.read_csv(INPUT_CSV, index_col=0)

print("Original shape:", df.shape)

# Convert probabilities to numeric
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# ==========================================
# REMOVE COLUMNS THAT ARE ALL ZERO
# ==========================================
df_nonzero = df.loc[:, (df != 0).any(axis=0)]

print("Shape after removing zero columns:", df_nonzero.shape)

# ==========================================
# SAVE MAPPING BEFORE RENAMING
# ==========================================
mapping = pd.DataFrame({
    "New_File_ID": range(len(df_nonzero.columns)),
    "Original_File_ID": df_nonzero.columns
})

mapping.to_csv(MAPPING_CSV, index=False)

# ==========================================
# RENAME FILE IDS FROM 0 TO N-1
# ==========================================
df_nonzero.columns = [str(i) for i in range(len(df_nonzero.columns))]

# ==========================================
# SAVE OUTPUT
# ==========================================
df_nonzero.to_csv(OUTPUT_CSV)

print("Done.")
print("Final shape:", df_nonzero.shape)
print("Files renamed from 0 to", len(df_nonzero.columns) - 1)
print("Output saved to:", OUTPUT_CSV)
print("Mapping saved to:", MAPPING_CSV)