import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Preprocessing script started...")

# ==============================
# Paths
# ==============================

DATA_DIR = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Data\random_subset\200"  # Change this to your subset folder

ratings_path = os.path.join(DATA_DIR, "ratings.csv")

train_output_path = os.path.join(DATA_DIR, "train.txt")
test_output_path = os.path.join(DATA_DIR, "test.txt")


# ==============================
# Load ratings
# ==============================

rating_df = pd.read_csv(ratings_path)

print("Original data:")
print(rating_df.head())


# ==============================
# Keep positive interactions only
# ==============================
# For MovieLens, rating >= 4 means the user liked the movie

#rating_df = rating_df[rating_df["rating"] >= 4].copy()


# ==============================
# Encode userId and movieId
# ==============================

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

rating_df["user_idx"] = user_encoder.fit_transform(rating_df["userId"])
rating_df["movie_idx"] = movie_encoder.fit_transform(rating_df["movieId"])

num_users = rating_df["user_idx"].nunique()
num_movies = rating_df["movie_idx"].nunique()
num_interactions = len(rating_df)

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Number of positive interactions: {num_interactions}")


# ==============================
# Train / Test split per user
# ==============================

train_lines = []
test_lines = []

for user_id, group in rating_df.groupby("user_idx"):

    items = group["movie_idx"].tolist()

    # Skip users with too few interactions
    if len(items) < 2:
        continue

    train_items, test_items = train_test_split(
        items,
        test_size=0.2,
        random_state=42
    )

    train_line = str(user_id) + " " + " ".join(map(str, train_items))
    test_line = str(user_id) + " " + " ".join(map(str, test_items))

    train_lines.append(train_line)
    test_lines.append(test_line)


# ==============================
# Save LightGCN format
# ==============================

with open(train_output_path, "w") as f:
    f.write("\n".join(train_lines))

with open(test_output_path, "w") as f:
    f.write("\n".join(test_lines))

print("Preprocessing finished.")
print(f"Saved train file to: {train_output_path}")
print(f"Saved test file to: {test_output_path}")