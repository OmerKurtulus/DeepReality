"""
DeepReality — Native XGBoost Booster Evaluator
==============================================

Evaluates a saved XGBoost JSON model without importing the xgboost
runtime.

Why this exists. XGBoost's macOS wheels link against Homebrew's OpenMP
runtime while PyTorch bundles its own; loading both into one process and
entering a parallel region segfaults the interpreter. Since every pin in
this system imports torch, the ensemble stage would crash on every
prediction. The available workarounds all constrain the wider process —
forcing OMP_NUM_THREADS=1 fixes the crash but also caps PyTorch's CPU
parallelism for unrelated work.

The dependency is unnecessary in the first place. A gradient-boosted
tree ensemble is not a weight matrix requiring optimised kernels; it is a
few hundred shallow decision trees, and evaluating them for a single row
is a few thousand comparisons. Traversing the saved JSON directly removes
the conflict entirely, makes inference independent of any native library,
and leaves xgboost needed only at training time.

Correctness is asserted rather than assumed: `tests/test_booster_eval.py`
compares this implementation against xgboost's own predictions.
"""

import json
import math
from pathlib import Path


class NativeBooster:
    """
    Minimal reader and evaluator for an XGBoost JSON model.

    Supports the numeric binary:logistic models this project trains.
    Categorical splits and multi-class objectives are rejected explicitly
    rather than mis-evaluated, since silently wrong scores are worse than
    an error.
    """

    def __init__(self, model_path: str | Path):
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))
        learner = model["learner"]

        objective = learner["objective"]["name"]
        if objective != "binary:logistic":
            raise ValueError(
                f"NativeBooster supports binary:logistic, got '{objective}'"
            )

        params = learner["learner_model_param"]
        self.num_feature = int(params["num_feature"])
        # base_score is serialised as a bracketed list, e.g. "[5E-1]"
        self.base_score = float(params["base_score"].strip("[]"))

        self.feature_names = learner.get("feature_names") or []

        booster_model = learner["gradient_booster"]["model"]
        self.trees = []
        for tree in booster_model["trees"]:
            if any(int(s) != 0 for s in tree.get("split_type", [])):
                raise ValueError("Categorical splits are not supported")
            self.trees.append({
                "left": tree["left_children"],
                "right": tree["right_children"],
                "index": tree["split_indices"],
                "cond": tree["split_conditions"],
                "default_left": tree["default_left"],
                "weight": tree["base_weights"],
            })

    def _leaf_value(self, tree: dict, row: list[float]) -> float:
        """Walk one tree to its leaf and return that leaf's contribution."""
        node = 0
        left, right = tree["left"], tree["right"]
        while left[node] != -1:
            value = row[tree["index"][node]]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                # A missing value follows the direction learned for it,
                # which is why absent evidence must be NaN and not zero.
                node = left[node] if tree["default_left"][node] else right[node]
            elif value < tree["cond"][node]:
                node = left[node]
            else:
                node = right[node]
        return tree["weight"][node]

    def margin(self, row: list[float]) -> float:
        """Raw score before the logistic link, including the base score."""
        if len(row) != self.num_feature:
            raise ValueError(
                f"Model expects {self.num_feature} features, received {len(row)}"
            )
        # boost_from_average stores base_score already in probability space
        # for this objective, so it is mapped back to the margin.
        base = self.base_score
        base = min(max(base, 1e-7), 1 - 1e-7)
        total = math.log(base / (1.0 - base))
        for tree in self.trees:
            total += self._leaf_value(tree, row)
        return total

    def predict(self, row: list[float]) -> float:
        """Probability of the positive class for a single feature row."""
        z = max(-60.0, min(60.0, self.margin(row)))
        return 1.0 / (1.0 + math.exp(-z))

    @property
    def n_trees(self) -> int:
        return len(self.trees)
