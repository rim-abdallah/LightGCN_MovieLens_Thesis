import os
import pandas as pd

# ====== paths ======
DATASET_PATH = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Data"
BASE_OUTPUT_PATH = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Data\random_subset"

ratings_file = os.path.join(DATASET_PATH, "ratings.csv")
movies_file = os.path.join(DATASET_PATH, "movies.csv")

# ====== ask k ======
k = int(input("Enter number of movies to keep: "))

# ====== create unique output folder ======
folder_name = str(k)
output_folder = os.path.join(BASE_OUTPUT_PATH, folder_name)

counter = 1
while os.path.exists(output_folder):
    output_folder = os.path.join(BASE_OUTPUT_PATH, f"{k}_{counter}")
    counter += 1

os.makedirs(output_folder)

# ====== read MovieLens ======
ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

# ====== choose k movies randomly ======
all_movie_ids = movies["movieId"].unique()

if k > len(all_movie_ids):
    raise ValueError(
        f"k is bigger than number of movies. Max = {len(all_movie_ids)}"
    )

selected_movie_ids = (
    pd.Series(all_movie_ids)
    .sample(n=k)
    .tolist()
)

# ====== filter ratings/interactions ======
filtered_ratings = ratings[
    ratings["movieId"].isin(selected_movie_ids)
].copy()

filtered_movies = movies[
    movies["movieId"].isin(selected_movie_ids)
].copy()

# ====== save new dataset ======
filtered_ratings.to_csv(
    os.path.join(output_folder, "ratings.csv"),
    index=False
)

filtered_movies.to_csv(
    os.path.join(output_folder, "movies.csv"),
    index=False
)

print("Done!")
print(f"Selected movies: {k}")
print(f"Number of interactions kept: {len(filtered_ratings)}")
print(f"Output folder: {output_folder}")