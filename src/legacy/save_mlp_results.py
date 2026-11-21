import csv
import os

results = [
    (1, 0.3254, 0.1637),
    (2, 0.4525, 0.3842),
    (3, 0.5938, 0.5715),
    (4, 0.5483, 0.5384),
    (5, 0.4949, 0.4266),
    (6, 0.5277, 0.4334),
    (7, 0.3575, 0.2291),
    (8, 0.4731, 0.3942),
    (9, 0.4105, 0.2989),
    (10, 0.4675, 0.4476),
    (11, 0.8899, 0.8902),
    (12, 0.3987, 0.3119),
    (13, 0.3808, 0.2656),
    (14, 0.5297, 0.4352),
    (15, 0.7138, 0.7015),
]

output_dir = "results/tables"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "mlp_loso_results.csv"
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
        0.5043,
        0.4328
    ])

    writer.writerow([
        "std",
        0.1409,
        0.1808
    ])

print(f"Saved results to: {output_file}")
