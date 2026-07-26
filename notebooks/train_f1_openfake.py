# =============================================================================
#  PIN-F1 — retrain on OpenFake
#
#  Run in the SAME session as the extraction notebook: it reuses the helpers
#  and the already-loaded models defined there.
#
#  Why this corpus. The diagnosis measured the base detectors on both
#  available corpora and found them at chance on Hemg (0.51-0.57, with B3
#  inverted at 0.41) but carrying real signal on OpenFake (B2 alone 0.84,
#  consensus 0.73). A stacked ensemble cannot manufacture signal its base
#  learners lack, so fitting on Hemg produced a model that learned corpus
#  framing instead of forensic evidence. Fitting where the detectors work is
#  the correct response.
#
#  Split discipline. Training samples come from OpenFake's TRAIN split; the
#  800 already extracted came from its TEST split and are reserved,
#  untouched, as the held-out set. No sample appears in both.
# =============================================================================

N_OPENFAKE_TRAIN = 1600     # from the train split; ~23 min at observed rates

import os, json, time, numpy as np, pandas as pd

ART = "/content/drive/MyDrive/DeepReality/artifacts"
CKPT = f"{ART}/f1_features_v3_openfake_train.parquet"

# ---------------------------------------------------------------------------
print("=" * 78); print("  1 · SAMPLING OpenFake TRAIN SPLIT"); print("=" * 78)
# ---------------------------------------------------------------------------
if os.path.exists(CKPT) and len(pd.read_parquet(CKPT)) >= N_OPENFAKE_TRAIN:
    train_df = pd.read_parquet(CKPT)
    print(f"  checkpoint reused: {train_df.shape}")
else:
    rows = take_balanced(
        HOLDOUT_DATASET, N_OPENFAKE_TRAIN, "label",
        lambda v: 1 if str(v).lower() in ("fake", "1", "true") else 0,
        split="train", cfg=HOLDOUT_CONFIG)
    train_df = extract(rows, "openfake_train")

test_df = pd.read_parquet(f"{ART}/f1_features_v3_holdout.parquet")
hemg_df = pd.read_parquet(f"{ART}/f1_features_v3_train.parquet")
print(f"  train (OpenFake/train): {train_df.shape}")
print(f"  test  (OpenFake/test) : {test_df.shape}   — never seen in training")
print(f"  stress (Hemg)         : {hemg_df.shape}   — different distribution")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78); print("  2 · TRAINING"); print("=" * 78)
# ---------------------------------------------------------------------------
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             brier_score_loss, confusion_matrix)
from layer6_ensemble.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from config.settings import ENSEMBLE_CONFIG

TP, SEED = ENSEMBLE_CONFIG["training"], ENSEMBLE_CONFIG["training"]["random_state"]
FEATS = list(FEATURE_NAMES)

X = train_df[FEATS].to_numpy(np.float32)
y = train_df["__label__"].to_numpy(np.int32)
print(f"  design matrix {X.shape}   positives {int(y.sum())}/{len(y)}")

const = [FEATS[i] for i in range(X.shape[1]) if np.nanstd(X[:, i]) < 1e-9]
print(f"  zero-variance ({len(const)}): {const}")

# Calibration is fitted on a slice the booster never trained on. Calibrating
# on training scores yields a sigmoid fitted to memorised outputs and reports
# excellent, meaningless reliability.
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
              max_depth=TP["max_depth"], learning_rate=TP["learning_rate"],
              subsample=TP["subsample"], colsample_bytree=TP["colsample_bytree"],
              min_child_weight=TP["min_child_weight"],
              reg_lambda=TP["reg_lambda"], random_state=SEED)

skf = StratifiedKFold(n_splits=TP["cv_folds"], shuffle=True, random_state=SEED)
cv = []
for tr, va in skf.split(X_fit, y_fit):
    m = xgb.XGBClassifier(n_estimators=TP["n_estimators"],
                          early_stopping_rounds=TP["early_stopping_rounds"], **params)
    m.fit(X_fit[tr], y_fit[tr], eval_set=[(X_fit[va], y_fit[va])], verbose=False)
    cv.append(roc_auc_score(y_fit[va], m.predict_proba(X_fit[va])[:, 1]))
print(f"  {TP['cv_folds']}-fold CV ROC-AUC : {np.mean(cv):.4f} ± {np.std(cv):.4f}")

model = xgb.XGBClassifier(n_estimators=TP["n_estimators"],
                          early_stopping_rounds=TP["early_stopping_rounds"], **params)
model.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], verbose=False)
print(f"  boosting rounds used : {model.best_iteration + 1}")

raw_cal = model.predict_proba(X_cal)[:, 1]
pl = LogisticRegression(C=1e6).fit(raw_cal.reshape(-1, 1), y_cal)
A, B = float(pl.coef_[0][0]), float(pl.intercept_[0])
cal = lambda r: 1.0 / (1.0 + np.exp(-np.clip(A * r + B, -60, 60)))
print(f"  Platt a={A:.4f} b={B:.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78); print("  3 · EVALUATION"); print("=" * 78)
# ---------------------------------------------------------------------------
def ece(y_true, p, bins=10):
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(tot)

def report(name, df):
    Xe = df[FEATS].to_numpy(np.float32)
    ye = df["__label__"].to_numpy(np.int32)
    p = cal(model.predict_proba(Xe)[:, 1])
    pred = (p >= 0.5).astype(int)
    met = dict(roc_auc=round(float(roc_auc_score(ye, p)), 4),
               accuracy=round(float(accuracy_score(ye, pred)), 4),
               f1=round(float(f1_score(ye, pred)), 4),
               brier=round(float(brier_score_loss(ye, p)), 4),
               ece=round(ece(ye, p), 4), n=int(len(ye)))
    tn, fp, fn, tp = confusion_matrix(ye, pred).ravel()
    print(f"\n  {name}  (n={met['n']})")
    print(f"    ROC-AUC {met['roc_auc']}  acc {met['accuracy']}  F1 {met['f1']}"
          f"  Brier {met['brier']}  ECE {met['ece']}")
    print(f"    TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    return met

held = report("HELD-OUT — OpenFake test split", test_df)
stress = report("STRESS — Hemg (different distribution)", hemg_df)

print("\n  Per-detector ROC-AUC on the held-out split, as the bar to clear:")
for f in ("b1_prob", "b2_prob", "b3_prob", "b4_fake_score", "consensus_mean"):
    col = test_df[f].to_numpy(np.float64); ye = test_df["__label__"].to_numpy()
    ok = ~np.isnan(col)
    if ok.sum() > 10 and len(set(ye[ok])) > 1:
        print(f"    {f:<18} {roc_auc_score(ye[ok], col[ok]):.4f}")
print(f"    {'PIN-F1':<18} {held['roc_auc']:.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78); print("  4 · FEATURE IMPORTANCE"); print("=" * 78)
# ---------------------------------------------------------------------------
booster = model.get_booster(); booster.feature_names = FEATS
gain = booster.get_score(importance_type="gain")
importance = {f: float(gain.get(f, 0.0)) for f in FEATS}
for f, g in sorted(importance.items(), key=lambda kv: -kv[1])[:12]:
    if g > 0:
        print(f"    {f:<34} {g:10.2f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78); print("  5 · DECISION"); print("=" * 78)
# ---------------------------------------------------------------------------
best_detector = max(
    roc_auc_score(test_df["__label__"], test_df[f].fillna(0.5))
    for f in ("b1_prob", "b2_prob", "b3_prob", "b4_fake_score"))

if held["roc_auc"] > best_detector and held["ece"] < 0.15:
    booster.save_model(f"{ART}/pin_f1_xgboost.json")
    json.dump({
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATS,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_dataset": f"{HOLDOUT_DATASET} [train split]",
        "holdout_dataset": f"{HOLDOUT_DATASET} [test split]",
        "training_samples": int(len(X_fit)),
        "calibration_samples": int(len(X_cal)),
        "label_convention": "1 = synthetic, 0 = authentic",
        "calibration": {"method": "platt", "a": A, "b": B},
        "cv_roc_auc_mean": round(float(np.mean(cv)), 4),
        "cv_roc_auc_std": round(float(np.std(cv)), 4),
        "holdout_metrics": held,
        "cross_corpus_stress": {"dataset": "Hemg/deepfake-and-real-images", **stress},
        "best_single_detector_roc_auc": round(float(best_detector), 4),
        "feature_importance": importance,
        "zero_variance_features": const,
        "evaluation_kind": "held_out_same_corpus",
        "caveat": (
            "Held-out figures are measured on a disjoint split of the same "
            "corpus. The cross-corpus stress result against Hemg is reported "
            "separately and is substantially weaker; the base detectors are "
            "at chance on that corpus, so the ensemble cannot recover there."
        ),
    }, open(f"{ART}/pin_f1_metadata.json", "w"), indent=2)
    print(f"""
  PIN-F1 {held['roc_auc']:.4f} clears the best single detector
  ({best_detector:.4f}) and is adequately calibrated (ECE {held['ece']}).
  Saved to {ART}/

  Download pin_f1_xgboost.json and pin_f1_metadata.json into models/""")
else:
    print(f"""
  PIN-F1 {held['roc_auc']:.4f} does not clear the best single detector
  ({best_detector:.4f}), or calibration is inadequate (ECE {held['ece']}).
  Not saved. A fusion stage that fails to beat its own best input adds
  nothing but complexity, and saying so is the correct outcome.""")
