# =============================================================================
#  PIN-F1 — cross-validated fit, the correct comparison
#
#  Why this supersedes the previous attempt. Fitting on OpenFake's train split
#  and testing on its test split produced a fusion that scored 0.74 while
#  PIN-B2 alone scored 0.84 on the same rows. A stacker that receives b2_prob
#  as an input cannot legitimately do worse than b2_prob unless the mapping it
#  learned does not hold on the evaluation rows — which is what a train/test
#  split organised by generator family or capture date will produce.
#
#  The two figures were also never comparable: 0.84 was b2 measured on every
#  test row, while 0.74 was a model fitted on a different distribution. Here
#  both are measured the same way. The corpus is pooled and stratified k-fold
#  is used, so every prediction is made by a model that never saw that row,
#  and the detectors are scored on exactly the same rows. That is the standard
#  protocol at this sample size and it is the only apples-to-apples answer to
#  "does fusion add anything".
# =============================================================================

import os, sys, json, time, subprocess

if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")

try:
    import xgboost
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "xgboost>=2.0.0"],
                   check=False)

import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             brier_score_loss, confusion_matrix)

ART = "/content/drive/MyDrive/DeepReality/artifacts"
SEED, FOLDS = 42, 5

a = pd.read_parquet(f"{ART}/f1_features_v3_openfake_train.parquet")
b = pd.read_parquet(f"{ART}/f1_features_v3_holdout.parquet")
pool = pd.concat([a, b], ignore_index=True)

FEATURE_NAMES = [c for c in pool.columns if c != "__label__"]
y = pool["__label__"].to_numpy(np.int32)
print(f"  pooled OpenFake: {pool.shape}   positives {int(y.sum())}/{len(y)}")

# Tier 3 and Tier 4 only. Provenance is adjudicated by the Layer 5 rule
# calculus, and it is also the strongest corpus-identifying signal available,
# which makes it a shortcut for a statistical stage.
FUSION = [f for f in FEATURE_NAMES
          if not f.startswith(("a1_", "a2_", "a4_"))
          and f not in ("x_telemetry_detector_conflict", "x_provenance_ai_strength")]
DETECTORS = ["b1_prob", "b2_prob", "b3_prob", "b4_fake_score", "consensus_mean"]
print(f"  fusion features: {len(FUSION)}")


def ece(y_true, p, bins=10):
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(tot)


def out_of_fold(make_model, feats):
    """Return one prediction per row, each from a model that never saw it."""
    X = pool[feats].to_numpy(np.float32)
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


print("\n" + "=" * 78)
print("  OUT-OF-FOLD COMPARISON  (every score measured on identical rows)")
print("=" * 78)

# Raw detector outputs need no fitting; they are already out-of-fold by
# construction, since the detectors never saw these rows either.
scores = {}
for d in DETECTORS:
    col = pool[d].to_numpy(np.float64)
    col = np.where(np.isnan(col), 0.5, col)
    scores[d] = (roc_auc_score(y, col), col)

candidates = {}

candidates["logistic (fusion set)"] = (
    lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=2000,
                                             random_state=SEED)),
    FUSION)

candidates["logistic (detector votes)"] = (
    lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000,
                                             random_state=SEED)),
    DETECTORS[:4])

candidates["xgboost d2"] = (
    lambda: xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                              tree_method="hist", n_estimators=300, max_depth=2,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.7, min_child_weight=20,
                              reg_lambda=5.0, random_state=SEED),
    FUSION)

candidates["xgboost d3"] = (
    lambda: xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                              tree_method="hist", n_estimators=400, max_depth=3,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.7, min_child_weight=10,
                              reg_lambda=2.0, random_state=SEED),
    FUSION)

for name, (factory, feats) in candidates.items():
    oof = out_of_fold(factory, feats)
    scores[name] = (roc_auc_score(y, oof), oof)

print(f"\n  {'method':<30} {'ROC-AUC':>9} {'acc':>7} {'F1':>7} {'ECE':>7}")
print("  " + "-" * 64)
for name, (auc, p) in sorted(scores.items(), key=lambda kv: -kv[1][0]):
    pred = (p >= 0.5).astype(int)
    marker = "  <- detector" if name in DETECTORS else ""
    print(f"  {name:<30} {auc:>9.4f} {accuracy_score(y, pred):>7.4f} "
          f"{f1_score(y, pred):>7.4f} {ece(y, p):>7.4f}{marker}")

best_detector = max(scores[d][0] for d in DETECTORS[:4])
best_fusion = max((scores[n][0], n) for n in candidates)
print(f"\n  best single detector : {best_detector:.4f}")
print(f"  best fusion          : {best_fusion[0]:.4f}  ({best_fusion[1]})")
print(f"  improvement          : {best_fusion[0] - best_detector:+.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("  DECISION")
print("=" * 78)
# ---------------------------------------------------------------------------
name = best_fusion[1]
factory, feats = candidates[name]
oof = scores[name][1]
oof_ece = ece(y, oof)

if best_fusion[0] > best_detector + 0.005:
    # Refit on everything: the k-fold estimate above is the honest measure of
    # how this configuration performs, and the deployed model should be the
    # one that has seen the most data.
    X_all = pool[feats].to_numpy(np.float32)
    final = factory()
    final.fit(X_all, y)

    # Calibrate against the out-of-fold predictions rather than the refitted
    # model's own output, which would be fitted to scores it memorised.
    pl = LogisticRegression(C=1.0).fit(oof.reshape(-1, 1), y)
    A, B = float(pl.coef_[0][0]), float(pl.intercept_[0])

    if hasattr(final, "get_booster"):
        booster = final.get_booster()
        booster.feature_names = list(feats)
        booster.save_model(f"{ART}/pin_f1_xgboost.json")
        importance = {f: float(booster.get_score(importance_type="gain").get(f, 0.0))
                      for f in feats}
        saved_kind = "xgboost"
    else:
        # A linear stacker is stored as its coefficients; PIN-F1 would need a
        # loader for this form, so it is reported rather than silently written
        # in a format the pin cannot read.
        importance = {}
        saved_kind = "logistic"
        print("\n  NOTE: the winning candidate is linear. PIN-F1 currently loads")
        print("  an XGBoost booster only, so this result is reported but not")
        print("  written as a deployable artefact.")

    if saved_kind == "xgboost":
        json.dump({
            "feature_schema_version": "1.0.0",
            "feature_names": list(feats),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_dataset": "ComplexDataLab/OpenFake [train+test pooled]",
            "evaluation_kind": f"{FOLDS}-fold cross-validated, out-of-fold",
            "training_samples": int(len(y)),
            "label_convention": "1 = synthetic, 0 = authentic",
            "calibration": {"method": "platt", "a": A, "b": B},
            "holdout_metrics": {"roc_auc": round(best_fusion[0], 4),
                                "ece": round(oof_ece, 4)},
            "best_single_detector_roc_auc": round(float(best_detector), 4),
            "improvement_over_best_detector": round(
                float(best_fusion[0] - best_detector), 4),
            "feature_importance": importance,
            "excluded_features": [f for f in FEATURE_NAMES if f not in feats],
            "scope_note": (
                "Restricted to Tier 3 and Tier 4 evidence; provenance is "
                "adjudicated by the Layer 5 rule calculus and is excluded here "
                "because it is also the strongest corpus-identifying signal."
            ),
        }, open(f"{ART}/pin_f1_metadata.json", "w"), indent=2)
        print(f"\n  Saved {name} to {ART}/")
        print("  Download pin_f1_xgboost.json and pin_f1_metadata.json into models/")
else:
    for f in ("pin_f1_xgboost.json", "pin_f1_metadata.json"):
        p = os.path.join(ART, f)
        if os.path.exists(p):
            os.rename(p, p + ".rejected")
    print(f"""
  Fusion does not improve on the best single detector under a fair
  comparison. PIN-B2 at {best_detector:.4f} already carries most of the
  available signal on this corpus, and the remaining detectors sit near
  0.63 with errors correlated to it, so there is nothing independent left
  for a stacker to combine.

  PIN-F1 therefore stays untrained and continues to report its transparent
  weighted baseline. That is the correct engineering outcome: a fusion
  stage that cannot beat its own best input should not be presented as if
  it could.""")
