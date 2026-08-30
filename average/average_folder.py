from pathlib import Path
import pandas as pd
import csv
import re


# ============================================================
# FOLDER
# ============================================================

FOLDER = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\result_alpha\com alpha"
)

OUTPUT_FILE = FOLDER / "com_alpha_averages.csv"


# ============================================================
# OUTPUT ORDER
# ============================================================

ALPHAS = [0.25, 0.5, 0.75]
COMMUNITIES = [2, 3, 4, 5, 10]


# ============================================================
# COLUMNS NOT TO AVERAGE
# ============================================================

IGNORE_COLUMNS = {
    "Run",
    "run",
    "Run_ID",
    "run_id",
    "RunID",
    "runid"
}


# ============================================================
# READ FILES
# ============================================================

results = {}

for csv_file in FOLDER.glob("*.csv"):

    # Do not read the output file again
    if csv_file.name == OUTPUT_FILE.name:
        continue

    # Example:
    # com_3_alpha_0.5_com.csv
    match = re.match(
        r"com_(\d+)_alpha_([0-9.]+)_com\.csv$",
        csv_file.name,
        re.IGNORECASE
    )

    if not match:
        print(f"Skipping: {csv_file.name}")
        continue

    community = int(match.group(1))
    alpha = float(match.group(2))

    print()
    print("======================================")
    print(f"Reading: {csv_file.name}")
    print(f"Community = {community}")
    print(f"Alpha = {alpha}")

    # ========================================================
    # READ CSV
    # ========================================================

    df = pd.read_csv(
        csv_file,
        sep=None,
        engine="python"
    )

    # ========================================================
    # CLEAN COLUMN NAMES
    # ========================================================

    cleaned_columns = []

    for col in df.columns:

        col = str(col).strip()
        col = col.replace("\ufeff", "")

        # ----------------------------------------------------
        # IMPORTANT:
        # gain_100 -> gain
        # gain_99  -> gain
        # colors_100 -> colors
        # colors_99  -> colors
        # etc.
        # ----------------------------------------------------

        col = re.sub(r"_(99|100)$", "", col)

        cleaned_columns.append(col)

    df.columns = cleaned_columns

    print("Columns after cleaning:")
    print(df.columns.tolist())

    # ========================================================
    # CALCULATE AVERAGES
    # ========================================================

    averages = {}

    for column in df.columns:

        if column in IGNORE_COLUMNS:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if values.notna().any():
            averages[column] = values.mean()

    results[(community, alpha)] = averages

    print("Number of runs:", len(df))


# ============================================================
# CHECK
# ============================================================

if not results:
    raise RuntimeError(
        "No matching CSV files were found."
    )


# ============================================================
# GET ALL METRIC NAMES
# ============================================================

metric_columns = []

for averages in results.values():

    for column in averages:

        if column not in metric_columns:
            metric_columns.append(column)


# ============================================================
# CREATE OUTPUT CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow(["com"])
    writer.writerow([])

    # ========================================================
    # ALPHA SECTIONS
    # ========================================================

    for alpha in ALPHAS:

        writer.writerow([f"for alpha {alpha}"])

        # Header
        writer.writerow(
            ["com"] + metric_columns
        )

        # ====================================================
        # COMMUNITIES
        # ====================================================

        for community in COMMUNITIES:

            key = (community, alpha)

            if key not in results:

                print(
                    f"WARNING: Missing community "
                    f"{community}, alpha {alpha}"
                )

                writer.writerow([community])
                continue

            averages = results[key]

            row = [community]

            for column in metric_columns:

                if column in averages:
                    row.append(
                        round(averages[column], 6)
                    )
                else:
                    row.append("")

            writer.writerow(row)

        writer.writerow([])
        writer.writerow([])


# ============================================================
# DONE
# ============================================================

print()
print("==========================================")
print("DONE")
print("==========================================")
print("Output:")
print(OUTPUT_FILE)