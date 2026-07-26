"""
Assert that the dependency-free evaluator reproduces xgboost exactly.

The evaluator in layer6_ensemble.booster_eval exists to avoid loading the
xgboost runtime alongside PyTorch (see that module for why). A
reimplementation of someone else's inference is only acceptable if it is
verified against the original, so this compares both on random inputs
including the missing-value paths that the default_left flags govern.

Run:  python3 tests/test_booster_eval.py
It is skipped cleanly when no trained artefact is present.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL = PROJECT_ROOT / "models" / "pin_f1_xgboost.json"
META = PROJECT_ROOT / "models" / "pin_f1_metadata.json"
N_SAMPLES = 500
TOLERANCE = 1e-6

# xgboost must run in its own interpreter: importing it into a process that
# has already loaded PyTorch is exactly the crash this module works around.
REFERENCE_SCRIPT = """
import json, sys, numpy as np, xgboost as xgb
model_path, meta_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
feature_names = json.load(open(meta_path))["feature_names"]
booster = xgb.Booster(); booster.load_model(model_path)
rng = np.random.default_rng(0)
X = rng.normal(size=(n, len(feature_names))).astype(np.float32) * 3.0
X[rng.random(X.shape) < 0.25] = np.nan
preds = booster.predict(xgb.DMatrix(X, feature_names=feature_names))
json.dump({"X": np.where(np.isnan(X), None, X).tolist(),
           "y": preds.tolist()}, sys.stdout)
"""


def main() -> int:
    if not MODEL.exists() or not META.exists():
        print("SKIP: no trained PIN-F1 artefact in models/")
        return 0

    print("Generating reference predictions in an isolated interpreter...")
    result = subprocess.run(
        [sys.executable, "-c", REFERENCE_SCRIPT, str(MODEL), str(META),
         str(N_SAMPLES)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("FAIL: reference generation failed")
        print(result.stderr[-2000:])
        return 1

    reference = json.loads(result.stdout)
    rows, expected = reference["X"], reference["y"]

    from layer6_ensemble.booster_eval import NativeBooster

    booster = NativeBooster(MODEL)
    print(f"Native evaluator: {booster.n_trees} trees, "
          f"{booster.num_feature} features")

    worst, worst_at = 0.0, -1
    nan_rows = 0
    for i, (row, want) in enumerate(zip(rows, expected)):
        if any(v is None for v in row):
            nan_rows += 1
        got = booster.predict([float("nan") if v is None else v for v in row])
        delta = abs(got - want)
        if delta > worst:
            worst, worst_at = delta, i

    print(f"Compared {len(rows)} rows ({nan_rows} containing missing values)")
    print(f"Maximum absolute deviation: {worst:.3e} (row {worst_at})")

    if worst <= TOLERANCE:
        print(f"PASS: native evaluator matches xgboost within {TOLERANCE:g}")
        return 0

    print(f"FAIL: deviation exceeds {TOLERANCE:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
