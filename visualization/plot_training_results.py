import re
import os
import pandas as pd
import matplotlib.pyplot as plt

log_file = r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\training_log.txt"
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)


def read_log_file(path):
    encodings = ["utf-16", "utf-8", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Could not read file with common encodings.")


def smooth_curve(series, window=15):
    return series.rolling(window=window, center=True, min_periods=1).mean()


epochs = []
train_loss = []

test_epochs = []
test_loss = []
test_recall = []

lines = read_log_file(log_file)

for line in lines:
    m = re.search(r"Epoch\s+(\d+).*?train==\[(\d+\.\d+)", line)
    if m:
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))

    t = re.search(
        r"Epoch\s+(\d+).*?test==\[(\d+\.\d+).*?recall=\[(\d+\.\d+)\]",
        line
    )
    if t:
        test_epochs.append(int(t.group(1)))
        test_loss.append(float(t.group(2)))
        test_recall.append(float(t.group(3)))


df_train = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_loss
}).drop_duplicates(subset="epoch")

df_test = pd.DataFrame({
    "epoch": test_epochs,
    "test_loss": test_loss,
    "test_recall": test_recall
}).drop_duplicates(subset="epoch")

df_train["smooth_loss"] = smooth_curve(df_train["train_loss"], window=25)
df_test["smooth_recall"] = smooth_curve(df_test["test_recall"], window=5)

df_train.to_csv(os.path.join(output_folder, "parsed_train_loss.csv"), index=False)
df_test.to_csv(os.path.join(output_folder, "parsed_test_metrics.csv"), index=False)

print("Train data:")
print(df_train.head())

print("\nTest data:")
print(df_test.head())


# Smoothed training loss graph
plt.figure(figsize=(8, 5))
plt.plot(df_train["epoch"], df_train["smooth_loss"], linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_loss_smooth.png"), dpi=300)
plt.show()


# Smoothed recall graph
plt.figure(figsize=(8, 5))
plt.plot(df_test["epoch"], df_test["smooth_recall"], linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Test Recall over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "test_recall_smooth.png"), dpi=300)
plt.show()