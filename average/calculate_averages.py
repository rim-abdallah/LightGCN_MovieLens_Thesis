import csv
import math
import re
from pathlib import Path


# =====================================================
# INPUT / OUTPUT FILES
# =====================================================

INPUT_CSV = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\result\15\com_10_pref_15.csv"
)

OUTPUT_CSV = Path(
    r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\result\15\average\pref_10_15.csv"
)


# =====================================================
# METRIC ORDER IN THE OUTPUT
#
# Expected RunAll_3 columns:
# gain_100
# colors_100
# nodes_100
# TNoCache_100
# TUncoded_100
# TCoded_100
# OverallGain_100
# LocalCacheDistribution_100
# LocalCacheContribution_100
# MulticastGain_100
# MulticastContribution_100
# =====================================================

METRICS = [
    "gain",
    "colors",
    "nodes",
    "TNoCache",
    "TUncoded",
    "TCoded",
    "OverallGain",
    "LocalCacheDistribution",
    "LocalCacheContribution",
    "MulticastGain",
    "MulticastContribution",
]


# =====================================================
# CONVERT CSV VALUE TO FLOAT
# =====================================================

def convert_to_float(value):
    """
    Convert a CSV value to float.

    Empty values, NA, N/A, nan, and none are ignored.
    """

    if value is None:
        return None

    value = value.strip()

    if value.lower() in {"", "na", "n/a", "nan", "none"}:
        return None

    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except ValueError:
        return None


# =====================================================
# FORMAT THE AVERAGE
# =====================================================

def format_number(value):
    """
    Display integers without unnecessary decimal zeros.

    Examples:
        1000.0   -> 1000
        0.221000 -> 0.221
        33.2105  -> 33.2105
    """

    if value is None:
        return "NA"

    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))

    return f"{value:.6f}".rstrip("0").rstrip(".")


# =====================================================
# READ THE ORIGINAL CSV
# =====================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input CSV was not found:\n{INPUT_CSV}"
    )

with INPUT_CSV.open(
    mode="r",
    newline="",
    encoding="utf-8-sig"
) as input_file:

    reader = csv.DictReader(input_file)

    if reader.fieldnames is None:
        raise ValueError(
            "The input CSV does not contain a header."
        )

    # Remove accidental spaces from column names
    original_fieldnames = reader.fieldnames

    fieldnames = [
        column.strip()
        for column in original_fieldnames
    ]

    rows = []

    for original_row in reader:
        normalized_row = {}

        for key, value in original_row.items():
            if key is not None:
                normalized_row[key.strip()] = value

        rows.append(normalized_row)


if not rows:
    raise ValueError(
        "The input CSV contains no experiment rows."
    )


# =====================================================
# AUTOMATICALLY FIND USER NUMBERS
#
# Detects:
# gain_10, gain_20, gain_100, ...
# =====================================================

user_numbers = []

for column in fieldnames:
    match = re.fullmatch(r"gain_(\d+)", column)

    if match:
        user_numbers.append(
            int(match.group(1))
        )

user_numbers = sorted(set(user_numbers))

if not user_numbers:
    raise ValueError(
        "No user columns were detected.\n"
        "Expected columns such as gain_10, "
        "gain_20, gain_100, etc."
    )


# =====================================================
# CALCULATE AVERAGE OF EVERY METRIC
# =====================================================

averages = {}

for user in user_numbers:
    for metric in METRICS:

        column_name = f"{metric}_{user}"

        if column_name not in fieldnames:
            print(
                f"Warning: column '{column_name}' "
                f"does not exist."
            )

            averages[column_name] = None
            continue

        values = []

        for row in rows:
            number = convert_to_float(
                row.get(column_name)
            )

            if number is not None:
                values.append(number)

        if values:
            averages[column_name] = (
                math.fsum(values) / len(values)
            )
        else:
            averages[column_name] = None


# =====================================================
# CREATE THE OUTPUT CSV
#
# Each user has:
#   - one header row
#   - one average row
#   - one empty row
# =====================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_CSV.open(
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as output_file:

    writer = csv.writer(output_file)

    for user in user_numbers:

        headers = [
            f"{metric}_{user}"
            for metric in METRICS
        ]

        average_row = [
            format_number(
                averages.get(column_name)
            )
            for column_name in headers
        ]

        writer.writerow(headers)
        writer.writerow(average_row)

        # Empty row between different user blocks
        writer.writerow([])


# =====================================================
# DISPLAY SUMMARY
# =====================================================

print("Average calculation completed.")
print(f"Number of experiment runs read: {len(rows)}")
print(f"Users detected: {user_numbers}")
print(f"Number of metrics: {len(METRICS)}")
print(f"Output saved to:\n{OUTPUT_CSV}")


# import csv
# import math
# import re
# from pathlib import Path


# # =====================================================
# # INPUT / OUTPUT FILES
# # =====================================================

# INPUT_CSV = Path(
#     r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\experiment4\results_zipf_com_0_2.csv"
# )

# OUTPUT_CSV = Path(
#     r"C:\Users\User\Desktop\THESIS\LightGCN_movieLens\Shared\experiment4\res\average_zipf_com_0.csv"
# )


# # =====================================================
# # METRIC ORDER IN THE OUTPUT
# # =====================================================

# METRICS = [
#     "gain",
#     "colors",
#     "nodes",
#     "TNoCache",
#     "TUncoded",
#     "TCoded",
#     "OverallGain",
#     "LocalCacheDistribution",
#     "LocalCacheContribution",
# ]


# # =====================================================
# # CONVERT CSV VALUE TO FLOAT
# # =====================================================

# def convert_to_float(value):
#     """
#     Convert a CSV value to float.

#     Empty values, NA, N/A and nan are ignored.
#     """

#     if value is None:
#         return None

#     value = value.strip()

#     if value.lower() in {"", "na", "n/a", "nan", "none"}:
#         return None

#     try:
#         number = float(value)

#         if not math.isfinite(number):
#             return None

#         return number

#     except ValueError:
#         return None


# # =====================================================
# # FORMAT THE AVERAGE
# # =====================================================

# def format_number(value):
#     """
#     Display integers without decimal zeros.

#     Examples:
#         1000.0      -> 1000
#         0.221000    -> 0.221
#         33.210500   -> 33.2105
#     """

#     if value is None:
#         return "NA"

#     if abs(value - round(value)) < 1e-12:
#         return str(int(round(value)))

#     return f"{value:.6f}".rstrip("0").rstrip(".")


# # =====================================================
# # READ THE ORIGINAL CSV
# # =====================================================

# if not INPUT_CSV.exists():
#     raise FileNotFoundError(
#         f"Input CSV was not found:\n{INPUT_CSV}"
#     )

# with INPUT_CSV.open(
#     mode="r",
#     newline="",
#     encoding="utf-8-sig"
# ) as input_file:

#     reader = csv.DictReader(input_file)

#     if reader.fieldnames is None:
#         raise ValueError("The input CSV does not contain a header.")

#     fieldnames = [
#         column.strip()
#         for column in reader.fieldnames
#     ]

#     rows = list(reader)


# if not rows:
#     raise ValueError("The input CSV contains no experiment rows.")


# # =====================================================
# # AUTOMATICALLY FIND USER NUMBERS
# #
# # It detects:
# # gain_10, gain_20, gain_30, ...
# # =====================================================

# user_numbers = []

# for column in fieldnames:
#     match = re.fullmatch(r"gain_(\d+)", column)

#     if match:
#         user_numbers.append(
#             int(match.group(1))
#         )

# user_numbers = sorted(set(user_numbers))

# if not user_numbers:
#     raise ValueError(
#         "No user columns were detected.\n"
#         "Expected columns such as gain_10, gain_20, etc."
#     )


# # =====================================================
# # CALCULATE AVERAGE OF EVERY COLUMN
# # =====================================================

# averages = {}

# for user in user_numbers:
#     for metric in METRICS:

#         column_name = f"{metric}_{user}"

#         if column_name not in fieldnames:
#             print(
#                 f"Warning: column '{column_name}' "
#                 f"does not exist."
#             )

#             averages[column_name] = None
#             continue

#         values = []

#         for row in rows:
#             number = convert_to_float(
#                 row.get(column_name)
#             )

#             if number is not None:
#                 values.append(number)

#         if values:
#             averages[column_name] = (
#                 math.fsum(values) / len(values)
#             )
#         else:
#             averages[column_name] = None


# # =====================================================
# # CREATE THE OUTPUT CSV
# #
# # Each user has:
# #   - one header row
# #   - one average row
# #   - one empty row
# # =====================================================

# OUTPUT_CSV.parent.mkdir(
#     parents=True,
#     exist_ok=True
# )

# with OUTPUT_CSV.open(
#     mode="w",
#     newline="",
#     encoding="utf-8-sig"
# ) as output_file:

#     writer = csv.writer(output_file)

#     for user in user_numbers:

#         headers = [
#             f"{metric}_{user}"
#             for metric in METRICS
#         ]

#         average_row = [
#             format_number(
#                 averages.get(column_name)
#             )
#             for column_name in headers
#         ]

#         writer.writerow(headers)
#         writer.writerow(average_row)

#         # Empty row between user blocks
#         writer.writerow([])


# # =====================================================
# # DISPLAY SUMMARY
# # =====================================================

# print("Average calculation completed.")
# print(f"Number of experiment runs read: {len(rows)}")
# print(f"Users detected: {user_numbers}")
# print(f"Output saved to:\n{OUTPUT_CSV}")