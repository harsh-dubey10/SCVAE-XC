# CNN and prototypical-network (meta-learning) few-shot study — design

Status: started straight away at the user's request (2026-10-24); the plan may be edited after results, but the
prediction below was written before any run and stays as written.

## Question

Does (a) a spatial CNN on the electrode map, (b) a prototypical network trained episodically for few-shot use,
or (c) both together, beat the calibration study's honest numbers (NormFusion v3 fair: 64.6% with no
calibration, 61.7% with 1 clip per emotion, 66.2% with 2 clips per emotion; cross-culture average of 6
directions)?

## Protocol (identical to the calibration study)

- 6 cross-culture directions, seeds 12, 34, 56, 5 calibration pools (2 films per emotion) per test person,
  N in {0, 1, 2} clips per emotion, calibration from the person's first session, the same test windows for every
  N (all of the person's windows whose film is not in the pool). Reuses `film_pool` and `split` from
  `run_calibration.py`; the v3 base probabilities are the saved ones in `results/calibration/base`.
- Training uses the source culture only; target labels are read only on the calibration windows.
- All hyperparameters fixed in advance (no selection on any target data): embedding 64, Adam 1e-3, weight decay
  1e-4, dropout 0.3; classifier 30 epochs, batch 256, class-balanced cross-entropy; prototypical network 3000
  episodes.

## Models (2 x 2)

- Encoder: **MLP** (EEG 310 -> 256 -> 128, eye 33 -> 64, -> 64) or **CNN** (EEG as 5 bands x 9 x 9 electrode
  map, two 3x3 conv layers 32/64, max-pool, -> 128; same eye branch, -> 64).
- Training: **classifier** (cross-entropy over source windows) or **prototypical** (episodes: one source person,
  K in {1, 2} films per emotion from their first session as support, their windows of other films as query;
  softmax over -scale x squared distance, scale learned).

## Variants scored

- N = 0: `v3`, `mlp_clf`, `cnn_clf`.
- N >= 1: `v3+person` (calibration study's combined), `mlp_clf+person`, `cnn_clf+person`, `mlp_proto`,
  `cnn_proto`, `mlp_proto+v3`, `cnn_proto+v3`, `cnn_proto+cnn_clf` (all 0.5 / 0.5 averages, fixed in advance).
- Sanity check: `v3` and `v3+person` must reproduce 64.6 / 61.7 / 66.2.

## Pre-registered prediction (cross-culture average, accuracy %)

| Variant | N = 0 | N = 1 | N = 2 |
|---|---:|---:|---:|
| cnn_clf (+person) | 62 | 61 | 65 |
| mlp_proto | — | 58 | 61 |
| cnn_proto | — | 58 | 61 |
| cnn_proto+v3 | — | 65 | 68 |

Success: the best fixed variant beats `v3+person` by >= 3 points at N = 2 (>= 69.2%) and wins >= 4 of 6
directions. Expected: not met; the CNN adds 0 to -3 points and the prototypical network helps mainly at N = 1.
