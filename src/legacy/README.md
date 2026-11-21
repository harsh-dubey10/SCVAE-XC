# Legacy code (old protocol)

This is the first SCVAE implementation and its leave-one-subject-out pipeline.
It predates the strict session-wise reproduction in `src/models/scvae_reference.py`
and `src/experiments/train_scvae_reference.py`. Its results are not comparable
with the strict results, so do not report them together.

Run from the repository root, e.g. `python -m src.legacy.train_scvae_loso`.
