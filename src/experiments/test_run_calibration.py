"""Checks for run_calibration on small synthetic data.

    python -m src.experiments.test_run_calibration
"""

import numpy as np

from src.experiments.run_calibration import (FOLLOWUPS, VARIANTS, WEIGHTS, calibrated_proba, film_pool,
                                             learned_weight, person_probas, person_scores, split, FEATURES)

rng = np.random.default_rng(5)

# Two people, two sessions each, 4 films per emotion (ids 1..12); 10 windows per film per session.
xs, ys, clips, subjects, recordings = [], [], [], [], []
for person in (1, 2):
    for session in (1, 2):
        for film in range(1, 13):
            emotion = (film - 1) // 4
            x = rng.normal(size=(10, 343)).astype(np.float32)
            x[:, :5] += 2 * emotion
            xs.append(x); ys.append(np.full(10, emotion)); clips.append(np.full(10, film))
            subjects.append(np.full(10, person)); recordings.append(np.full(10, 1000 * session + person))
x, y, clips, subjects, recordings = (np.concatenate(v) for v in (xs, ys, clips, subjects, recordings))
base = rng.dirichlet(np.ones(3), size=len(y)).astype(np.float32)

# T2.1 pools: deterministic per (draw, person), 2 films per emotion, only films the person watched,
# different draws differ.
pool = film_pool(y, clips, subjects, person=1, draw=0)
assert pool == film_pool(y, clips, subjects, person=1, draw=0)
assert sorted(pool) == [0, 1, 2] and all(len(films) == 2 for films in pool.values())
assert all(set(films) <= set(range(4 * e + 1, 4 * e + 5)) for e, films in pool.items())
assert any(film_pool(y, clips, subjects, person=1, draw=d) != pool for d in range(1, 5))

for n in (1, 2):
    calib, test = split(y, clips, subjects, recordings, person=1, pool=pool, n=n)
    # T2.2 film-disjoint: no shared film; calibration from the person's first session and that person only.
    assert not set(clips[calib].tolist()) & set(clips[test].tolist())
    assert set(subjects[calib].tolist()) == {1} and set((recordings[calib] // 1000).tolist()) == {1}
    assert set(clips[calib].tolist()) == {f for films in pool.values() for f in films[:n]}
    assert set(subjects[test].tolist()) == {1}
# T2.3 identical test windows for every N.
tests = [split(y, clips, subjects, recordings, person=1, pool=pool, n=n)[1] for n in (0, 1, 2)]
assert all((t == tests[0]).all() for t in tests) and tests[0].sum() == 2 * 6 * 10  # 6 films x 2 sessions
# T2.4 same-film: shares films with the test windows by construction, but never the same windows.
calib_sf, test_sf = split(y, clips, subjects, recordings, person=1, pool=pool, n=1, same_film=True)
assert set(clips[calib_sf].tolist()) <= set(clips[tests[0]].tolist()) and calib_sf.any()
assert not (calib_sf & test_sf).any() and set((recordings[calib_sf] // 1000).tolist()) == {1}

# T2.5 N = 0 combined is the base model; combined = 0.5 base + 0.5 person, exactly.
calib, test = split(y, clips, subjects, recordings, person=1, pool=pool, n=1)
assert np.array_equal(calibrated_proba("combined", base, x, y, calib, test, n=0), base[test])
person = calibrated_proba("person_only", base, x, y, calib, test, n=1)
assert np.array_equal(calibrated_proba("combined", base, x, y, calib, test, n=1), 0.5 * base[test] + 0.5 * person)
# T2.6 modality variants use only their columns: changing the other columns changes nothing.
for variant, keep in (("combined_eeg", FEATURES["eeg"]), ("combined_eye", FEATURES["eye"])):
    other = x.copy()
    mask = np.ones(343, bool); mask[keep] = False
    other[:, mask] = rng.normal(size=(len(x), mask.sum()))
    assert np.array_equal(calibrated_proba(variant, base, x, y, calib, test, n=1),
                          calibrated_proba(variant, base, other, y, calib, test, n=1)), variant

# T2.7 test labels are only used to score: shuffling them changes no prediction of any variant.
shuffled = y.copy(); shuffled[test] = rng.permutation(y[test])
for variant in VARIANTS:
    if variant == "combined_samefilm":
        continue  # its calibration windows are themselves labelled; checked through split() above
    assert np.array_equal(calibrated_proba(variant, base, x, y, calib, test, n=1),
                          calibrated_proba(variant, base, x, shuffled, calib, test, n=1)), variant

# Follow-up B (per-session calibration): calibration windows from every session, never a test film; each
# session's test windows are predicted by the model of that session only.
calib_all, test_all = split(y, clips, subjects, recordings, person=1, pool=pool, n=1, all_sessions=True)
assert set((recordings[calib_all] // 1000).tolist()) == {1, 2} and (test_all == tests[0]).all()
assert not set(clips[calib_all].tolist()) & set(clips[test_all].tolist())
changed = y.copy()
changed[calib_all & (recordings // 1000 == 2)] = rng.integers(0, 3, (calib_all & (recordings // 1000 == 2)).sum())
s1 = (recordings[test_all] // 1000) == 1
assert np.array_equal(person_probas(x, y, recordings, calib_all, test_all, per_session=True)[s1],
                      person_probas(x, changed, recordings, calib_all, test_all, per_session=True)[s1])

# Follow-up A (learned weight): one of the grid; only calibration labels are read; a perfect base model gets
# weight 0 (ties go to the smaller weight), a useless base model gets a positive weight.
calib2, test2 = split(y, clips, subjects, recordings, person=1, pool=pool, n=2)
w = learned_weight(base, x, y, clips, recordings, calib2, pool, n=2, per_session=False)
assert w in WEIGHTS
shuffled2 = y.copy(); shuffled2[test2] = rng.permutation(y[test2])
assert learned_weight(base, x, shuffled2, clips, recordings, calib2, pool, n=2, per_session=False) == w
perfect = np.eye(3, dtype=np.float32)[y]
assert learned_weight(perfect, x, y, clips, recordings, calib2, pool, n=2, per_session=False) == 0.0
useless = np.full_like(base, 1 / 3)
assert learned_weight(useless, x, y, clips, recordings, calib2, pool, n=2, per_session=False) > 0.0

# Scoring one person: every variant at every N, scored on the same test windows.
scores = person_scores(base, x, y, clips, subjects, recordings, person=1, draw=0)
assert sorted(scores) == [0, 1, 2]
assert set(scores[0]) == {"base_only", "combined"}
assert set(scores[1]) == set(VARIANTS) | {"persession"}  # learned weights need 2 calibration films per emotion
assert set(scores[2]) == set(VARIANTS) | set(FOLLOWUPS)
assert scores[0]["combined"] == scores[0]["base_only"] == scores[1]["base_only"] == scores[2]["base_only"]
assert all(0 <= s["accuracy"] <= 1 and set(s["sessions"]) == {"1", "2"}
           for v, s in scores[2].items() if v != "combined_samefilm")
# Same-film at N = 2 here uses both non-pool films per emotion from session 1, so only session 2 is tested.
assert set(scores[2]["combined_samefilm"]["sessions"]) == {"2"}
print("run_calibration checks passed")
