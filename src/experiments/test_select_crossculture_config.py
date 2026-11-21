"""Checks for leave-target-culture-out configuration choice.

    .venv/bin/python -m src.experiments.test_select_crossculture_config
"""

import json
import tempfile
from pathlib import Path

from src.experiments.select_crossculture_config import CULTURES, choose, load

C, G, F = CULTURES
# "a" is best on the German<->French validation transfers; "b" is far better on the transfers
# into Chinese, which must not influence the choice for Chinese.
scores = {}
for config, validation, test in [("a", 60, 40), ("b", 50, 99)]:
    scores.update({(config, G, F): validation, (config, F, G): validation,
                   (config, G, C): test, (config, F, C): test})
assert choose(scores, ["a", "b"], C) == ("a", 60)
assert choose(scores, ["a", "b"], G) is None  # transfers into German were never run
del scores[("a", F, C)]
assert choose(scores, ["a", "b"], C) == ("b", 50)  # "a" incomplete on a test transfer

# load: grid layout; sval2 / sval3 merge into one configuration.
root = Path(tempfile.mkdtemp())
for target, source, k, accuracy in [(G, C, 3, 0.5), (G, F, 2, 0.7)]:
    run = root / target / f"strict__bn_source__sval{k}__subjnorm__from_{source}" / "session_all" / "seed_12"
    run.mkdir(parents=True)
    for subject in (1, 2):
        (run / f"subject_{subject:02d}.json").write_text(json.dumps({"test_subject": subject, "accuracy": accuracy}))
loaded = load(root, pattern="*/*")
assert loaded == {("strict__bn_source__sval__subjnorm", C, G): 50.0,
                  ("strict__bn_source__sval__subjnorm", F, G): 70.0}, loaded
print("Selection tests passed.")
