import csv
import os

results = [
    (1, 0.3254, 0.1637),
    (2, 0.5455, 0.5067),
    (3, 0.5962, 0.5813),
    (4, 0.6144, 0.6151),
    (5, 0.4731, 0.3903),
    (6, 0.4972, 0.4074),
    (7, 0.3456, 0.2063),
    (8, 0.5891, 0.4964),
    (9, 0.4537, 0.3717),
    (10, 0.3761, 0.2538),
    (11, 0.7961, 0.7939),
    (12, 0.4390, 0.3821),
    (13, 0.3511, 0.2325),
    (14, 0.5570, 0.5565),
    (15, 0.5760, 0.5398),
]

output_dir = "results/tables"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "svm_loso_results.csv"
)

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "subject",
        "accuracy",
        "macro_f1"
    ])

    writer.writerows(results)

    writer.writerow([])
    writer.writerow([
        "mean",
        0.5024,
        0.4332
    ])

    writer.writerow([
        "std",
        0.1231,
        0.1686
    ])

print(f"Saved results to: {output_file}")
