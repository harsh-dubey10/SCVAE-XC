import argparse
import glob
import torch


parser = argparse.ArgumentParser(description="Validate converted SEED tensor files.")
parser.add_argument("--data-dir", default="data/processed/SEED")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.data_dir}/*/*.pth"))

print("Files:", len(files))

total_samples = 0

for path in files:
    d = torch.load(path, weights_only=True)

    x = torch.stack(d["data"])
    y = torch.tensor(d["labels"])

    assert x.shape[1:] == (62, 5)
    assert len(x) == len(y)
    assert set(y.tolist()) <= {0, 1, 2}
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()

    total_samples += len(x)

print("Total samples:", total_samples)
print("All files passed verification.")
