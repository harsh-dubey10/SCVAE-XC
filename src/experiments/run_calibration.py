"""Score the calibration study from saved base probabilities.

For every base run (results/calibration/base/<target>/<mode>__from_<source>/seed_<s>.npz), every test person
and 5 random calibration pools (2 films per emotion), the person's windows are split into calibration windows
(first session, N films per emotion from the pool) and test windows (all of the person's windows whose film is
not in the pool — identical for every N). Variants:

  base_only          the base model (NormFusion v3 fair), no calibration
  person_only        a logistic regression (C = 0.1) on the person's calibration windows alone
  combined           mean of base and person probabilities (primary; N = 0 is the base model)
  combined_eeg/eye   the same with the person model on EEG (0:310) or eye (310:343) features only
  combined_samefilm  calibration from session-1 windows of films that are also tested (leaky; Q2 only)

Writes results/calibration/scores/<target>/<mode>__from_<source>/seed_<s>.json.

    python -m src.experiments.run_calibration --jobs 8

Design: docs/superpowers/specs/2026-10-04-calibration-design.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from src.baselines.norm_fusion import fit_lr, load_culture
from src.experiments.run_calibration_base import subset

FEATURES = {"all": slice(0, 343), "eeg": slice(0, 310), "eye": slice(310, 343)}
PERSON_C, PERSON_WEIGHT = 0.1, 0.5
POOL_PER_EMOTION, DRAWS, NS = 2, 5, (0, 1, 2)
VARIANTS = ["base_only", "person_only", "combined", "combined_eeg", "combined_eye", "combined_samefilm"]
# Follow-ups after the pre-registered result (exploratory): per-session calibration (every session's first
# pool films, one person model per session), a weight on the person model learned by leave-one-film-out on the
# calibration windows (needs 2 films per emotion), and both.
FOLLOWUPS = ["persession", "learned", "persession_learned"]
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def film_pool(y, clips, subjects, person, draw):
    """{emotion: films} — POOL_PER_EMOTION films per emotion the person watched, drawn with
    default_rng(1000 * draw + person)."""
    rng = np.random.default_rng(1000 * draw + int(person))
    mine = subjects == person
    return {e: sorted(rng.choice(np.unique(clips[mine & (y == e)]), POOL_PER_EMOTION, replace=False).tolist())
            for e in range(3)}


def split(y, clips, subjects, recordings, person, pool, n, same_film=False, all_sessions=False):
    """(calibration mask, test mask). Test: the person's windows whose film is not in the pool. Calibration:
    the person's first-session windows (every session's with all_sessions) of the first n pool films per
    emotion, or (same_film) of the first n test films per emotion, whose windows then leave the test set."""
    mine = subjects == person
    session = recordings // 1000
    first = mine if all_sessions else mine & (session == session[mine].min())
    test = mine & ~np.isin(clips, [f for films in pool.values() for f in films])
    if same_film:
        films = [f for e in range(3) for f in sorted(np.unique(clips[test & (y == e)]).tolist())[:n]]
        calib = first & np.isin(clips, films)
        return calib, test & ~calib
    return first & np.isin(clips, [f for films in pool.values() for f in films[:n]]), test


def calibrated_proba(variant, base, x, y, calib, evaluate, n):
    """Probabilities on the `evaluate` windows. Labels are read only on the calibration windows."""
    if variant == "base_only" or n == 0:
        return base[evaluate]
    features = FEATURES[{"combined_eeg": "eeg", "combined_eye": "eye"}.get(variant, "all")]
    person = fit_lr(x[calib][:, features], y[calib], PERSON_C).predict_proba(x[evaluate][:, features])
    if variant == "person_only":
        return person
    return (1 - PERSON_WEIGHT) * base[evaluate] + PERSON_WEIGHT * person


def person_probas(x, y, recordings, calib, evaluate, per_session):
    """Person-model probabilities on the `evaluate` windows: one model on all calibration windows, or one per
    session applied to that session's windows only."""
    if not per_session:
        return fit_lr(x[calib], y[calib], PERSON_C).predict_proba(x[evaluate])
    session = recordings // 1000
    out = np.empty((int(evaluate.sum()), 3))
    for s in np.unique(session[evaluate]):
        train = calib & (session == s)
        out[session[evaluate] == s] = fit_lr(x[train], y[train], PERSON_C).predict_proba(x[evaluate & (session == s)])
    return out


def learned_weight(base, x, y, clips, recordings, calib, pool, n, per_session):
    """Weight on the person model, chosen by leave-one-film-out on the calibration windows (one calibration film
    per emotion held out at a time); ties go to the smaller weight. Reads calibration labels only."""
    scores = {w: [] for w in WEIGHTS}
    for k in range(n):
        held = calib & np.isin(clips, [films[k] for films in pool.values()])
        person = person_probas(x, y, recordings, calib & ~held, held, per_session)
        for w in WEIGHTS:
            scores[w].append(float((((1 - w) * base[held] + w * person).argmax(axis=1) == y[held]).mean()))
    return max(WEIGHTS, key=lambda w: np.mean(scores[w]))


def _score(y, recordings, mask, proba):
    pred, true = proba.argmax(axis=1), y[mask]
    sessions = recordings[mask] // 1000
    return {"accuracy": float((pred == true).mean()), "macro_f1": float(f1_score(true, pred, average="macro")),
            "sessions": {str(s): float((pred[sessions == s] == true[sessions == s]).mean()) for s in np.unique(sessions)}}


def person_scores(base, x, y, clips, subjects, recordings, person, draw):
    """{N: {variant: scores}} for one person and one calibration pool."""
    pool = film_pool(y, clips, subjects, person, draw)
    out = {}
    for n in NS:
        calib, test = split(y, clips, subjects, recordings, person, pool, n)
        variants = ["base_only", "combined"] if n == 0 else VARIANTS
        out[n] = {}
        for variant in variants:
            if variant == "combined_samefilm":
                calib_sf, test_sf = split(y, clips, subjects, recordings, person, pool, n, same_film=True)
                out[n][variant] = _score(y, recordings, test_sf, calibrated_proba(variant, base, x, y, calib_sf,
                                                                                  test_sf, n))
            else:
                out[n][variant] = _score(y, recordings, test, calibrated_proba(variant, base, x, y, calib, test, n))
        if n == 0:
            continue
        calib_all, _ = split(y, clips, subjects, recordings, person, pool, n, all_sessions=True)
        for variant, mask, per_session in (("persession", calib_all, True), ("learned", calib, False),
                                           ("persession_learned", calib_all, True)):
            if variant != "persession" and n < 2:
                continue
            w = learned_weight(base, x, y, clips, recordings, mask, pool, n, per_session) \
                if variant != "persession" else PERSON_WEIGHT
            proba = (1 - w) * base[test] + w * person_probas(x, y, recordings, mask, test, per_session)
            out[n][variant] = {**_score(y, recordings, test, proba), "weight": w}
    return out


def score_run(path, n_jobs):
    data = np.load(path)
    target = path.parent.parent.name
    culture = load_culture(target)
    culture = subset(culture, np.unique(data["subjects"]))
    for field in ("y", "clips", "subjects", "recordings"):
        assert np.array_equal(getattr(culture, field), data[field]), f"{path}: windows do not line up ({field})"
    people = np.unique(data["subjects"])
    jobs = [(p, d) for p in people for d in range(DRAWS)]
    results = Parallel(n_jobs=n_jobs)(delayed(person_scores)(data["proba"], culture.x, culture.y, culture.clips,
                                                             culture.subjects, culture.recordings, p, d)
                                      for p, d in jobs)
    scores = {}
    for (p, d), result in zip(jobs, results):
        scores.setdefault(str(p), {})[str(d)] = {str(n): v for n, v in result.items()}
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-root", type=Path, default=Path("results/calibration/base"))
    parser.add_argument("--output-root", type=Path, default=Path("results/calibration/scores"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    for path in sorted(args.base_root.glob("*/*/seed_*.npz")):
        out = args.output_root / path.relative_to(args.base_root).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(score_run(path, args.jobs)))
        print(f"scored {path.relative_to(args.base_root)}", flush=True)


if __name__ == "__main__":
    main()
