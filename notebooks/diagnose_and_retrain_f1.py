# =============================================================================
#  PIN-F1 — diagnosis and retraining
#  Runs on the feature matrices already extracted to Drive. No image
#  processing, no model loading: this takes seconds, not minutes.
#
#  The first run produced ROC-AUC 0.50 on the cross-dataset holdout, which is
#  chance. This cell establishes why and whether a usable model exists in the
#  data we already have.
# =============================================================================

import os, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "/content/DeepReality")
from layer6_ensemble.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from config.settings import ENSEMBLE_CONFIG

import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             brier_score_loss, confusion_matrix)

ART = "/content/drive/MyDrive/DeepReality/artifacts"
TAG = "v3"
TP  = ENSEMBLE_CONFIG["training"]
SEED = TP["random_state"]

hemg = pd.read_parquet(f"{ART}/f1_features_{TAG}_train.parquet")
ofake = pd.read_parquet(f"{ART}/f1_features_{TAG}_holdout.parquet")
print(f"Hemg   {hemg.shape}   positives {int(hemg['__label__'].sum())}")
print(f"OpenFake {ofake.shape}  positives {int(ofake['__label__'].sum())}")

FEATS = list(FEATURE_NAMES)

def XY(df):
    return df[FEATS].to_numpy(np.float32), df["__label__"].to_numpy(np.int32)


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("  1 · DO THE BASE DETECTORS CARRY ANY SIGNAL?")
print("=" * 78)
# A stacked ensemble cannot manufacture signal its base learners lack. If the
# detectors are at chance on a corpus, anything the meta-learner achieves
# there is coming from somewhere else — which is the failure we are chasing.
# ---------------------------------------------------------------------------
DETECTORS = ["b1_prob", "b2_prob", "b3_prob", "b4_fake_score", "consensus_mean"]

print(f"\n  {'feature':<24} {'Hemg':>10} {'OpenFake':>10}")
print("  " + "-" * 46)
for f in DETECTORS:
    row = f"  {f:<24}"
    for df in (hemg, ofake):
        col, y = df[f].to_numpy(np.float64), df["__label__"].to_numpy()
        ok = ~np.isnan(col)
        auc = roc_auc_score(y[ok], col[ok]) if ok.sum() > 10 and len(set(y[ok])) > 1 else np.nan
        row += f" {auc:>10.4f}"
    print(row)

print("""
  Reading: 0.50 is chance. Below 0.50 means the detector is anti-correlated
  with the label on that corpus — its signal is present but inverted, which
  is worse than noise because a fusion model will learn to trust it backwards.
""")


# ---------------------------------------------------------------------------
print("=" * 78)
print("  2 · WHICH FEATURES SEPARATE THE TWO CORPORA?")
print("=" * 78)
# A feature that predicts which dataset a sample came from is a shortcut
# vector: a meta-learner will exploit it in place of forensic evidence, and
# the exploit evaporates the moment the distribution changes.
# ---------------------------------------------------------------------------
both = pd.concat([hemg.assign(__corpus__=0), ofake.assign(__corpus__=1)])
Xb = both[FEATS].to_numpy(np.float32)
yb = both["__corpus__"].to_numpy(np.int32)

corpus_auc = {}
for i, f in enumerate(FEATS):
    col = Xb[:, i].astype(np.float64)
    ok = ~np.isnan(col)
    if ok.sum() > 50 and len(set(yb[ok])) > 1 and np.nanstd(col) > 1e-9:
        corpus_auc[f] = abs(roc_auc_score(yb[ok], col[ok]) - 0.5) + 0.5

print("\n  Most corpus-identifying features (1.00 = perfectly separates them):")
for f, a in sorted(corpus_auc.items(), key=lambda kv: -kv[1])[:12]:
    flag = "  <-- shortcut risk" if a > 0.80 else ""
    print(f"    {f:<34} {a:.4f}{flag}")


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("  3 · CANDIDATE CONFIGURATIONS")
print("=" * 78)
# ---------------------------------------------------------------------------

# Composition features describe how a photograph was framed, not whether it
# was synthesised. They are the classic shortcut in curated corpora, where
# the two classes were assembled by different pipelines.
COMPOSITION = ["a4_face_count", "a4_max_face_confidence",
               "a4_max_face_area_ratio", "a4_max_sharpness"]
FORENSIC = [f for f in FEATS if f not in COMPOSITION]


def ece(y_true, p, bins=10):
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(total)


def fit_eval(name, train_df, test_df, feats, note=""):
    """Fit on one corpus, calibrate on a slice of it, evaluate on another."""
    Xtr_all = train_df[feats].to_numpy(np.float32)
    ytr_all = train_df["__label__"].to_numpy(np.int32)
    Xte = test_df[feats].to_numpy(np.float32)
    yte = test_df["__label__"].to_numpy(np.int32)

    Xf, Xc, yf, yc = train_test_split(
        Xtr_all, ytr_all, test_size=0.20, stratify=ytr_all, random_state=SEED)

    params = dict(objective="binary:logistic", eval_metric="auc",
                  tree_method="hist", max_depth=TP["max_depth"],
                  learning_rate=TP["learning_rate"], subsample=TP["subsample"],
                  colsample_bytree=TP["colsample_bytree"],
                  min_child_weight=TP["min_child_weight"],
                  reg_lambda=TP["reg_lambda"], random_state=SEED)

    m = xgb.XGBClassifier(n_estimators=TP["n_estimators"],
                          early_stopping_rounds=TP["early_stopping_rounds"],
                          **params)
    m.fit(Xf, yf, eval_set=[(Xc, yc)], verbose=False)

    raw_c = m.predict_proba(Xc)[:, 1]
    pl = LogisticRegression(C=1e6).fit(raw_c.reshape(-1, 1), yc)
    A, B = float(pl.coef_[0][0]), float(pl.intercept_[0])
    cal = lambda r: 1.0 / (1.0 + np.exp(-np.clip(A * r + B, -60, 60)))

    p_in = cal(raw_c)
    p_out = cal(m.predict_proba(Xte)[:, 1])
    pred_out = (p_out >= 0.5).astype(int)

    auc_in = roc_auc_score(yc, p_in)
    auc_out = roc_auc_score(yte, p_out)
    tn, fp, fn, tp = confusion_matrix(yte, pred_out).ravel()

    print(f"\n  [{name}] {note}")
    print(f"    in-distribution ROC-AUC : {auc_in:.4f}")
    print(f"    TRANSFER ROC-AUC        : {auc_out:.4f}   "
          f"acc {accuracy_score(yte, pred_out):.4f}  "
          f"F1 {f1_score(yte, pred_out):.4f}  ECE {ece(yte, p_out):.4f}")
    print(f"    TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    return dict(name=name, model=m, feats=feats, a=A, b=B,
                auc_in=float(auc_in), auc_out=float(auc_out),
                n_train=int(len(Xf)))


results = []
results.append(fit_eval("A", hemg, ofake, FEATS,
                        "train Hemg -> test OpenFake  (the original run)"))
results.append(fit_eval("B", ofake, hemg, FEATS,
                        "train OpenFake -> test Hemg  (roles reversed)"))
results.append(fit_eval("C", hemg, ofake, FORENSIC,
                        "train Hemg -> test OpenFake, composition features removed"))
results.append(fit_eval("D", ofake, hemg, FORENSIC,
                        "train OpenFake -> test Hemg, composition features removed"))

# Pooled: the model sees both distributions during training. This is not a
# transfer measurement and must never be reported as one, but it is the
# honest configuration to deploy when both corpora represent the operating
# domain and no third distribution is available to test on.
pool = pd.concat([hemg, ofake], ignore_index=True)
pool_tr, pool_te = train_test_split(
    pool, test_size=0.25, stratify=pool["__label__"], random_state=SEED)
results.append(fit_eval("E", pool_tr, pool_te, FORENSIC,
                        "pooled corpora, held-out split (NOT a transfer figure)"))


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("  4 · VERDICT")
print("=" * 78)
transfer = [r for r in results if r["name"] in ("A", "B", "C", "D")]
best = max(transfer, key=lambda r: r["auc_out"])
pooled = results[-1]

print(f"\n  Best transfer configuration : {best['name']}  "
      f"ROC-AUC {best['auc_out']:.4f}")
print(f"  Pooled configuration        : E  ROC-AUC {pooled['auc_out']:.4f}")

if best["auc_out"] >= 0.65:
    chosen, kind = best, "cross_dataset"
    print("\n  A configuration transfers. Saving it.")
elif pooled["auc_out"] >= 0.70:
    chosen, kind = pooled, "pooled_in_distribution"
    print("""
  No configuration transfers across corpora, but the pooled model is usable
  within the combined domain. Saving it, labelled honestly: its metric is an
  in-distribution estimate over both corpora, NOT evidence of generalisation
  to an unseen generator.""")
else:
    chosen, kind = None, None
    print("""
  Nothing reaches a usable standard. The base detectors carry too little
  signal on these corpora for a fusion stage to recover, so PIN-F1 should
  stay untrained and continue reporting its transparent baseline rather than
  present a fitted model that is no better than chance.""")

if chosen is not None:
    m, feats = chosen["model"], chosen["feats"]
    booster = m.get_booster(); booster.feature_names = list(feats)
    booster.save_model(f"{ART}/pin_f1_xgboost.json")
    gain = booster.get_score(importance_type="gain")
    meta = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(feats),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": chosen["name"],
        "evaluation_kind": kind,
        "training_samples": chosen["n_train"],
        "label_convention": "1 = synthetic, 0 = authentic",
        "calibration": {"method": "platt", "a": chosen["a"], "b": chosen["b"]},
        "holdout_metrics": {
            "roc_auc": round(chosen["auc_out"], 4),
            "in_distribution_roc_auc": round(chosen["auc_in"], 4),
        },
        "feature_importance": {f: float(gain.get(f, 0.0)) for f in feats},
        "excluded_features": [f for f in FEATS if f not in feats],
        "caveat": (
            "Base detectors measured at or near chance on both corpora; see "
            "section 1 of the diagnosis. This artefact should be retrained on "
            "a corpus where the detectors demonstrably carry signal."
        ),
    }
    with open(f"{ART}/pin_f1_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n  Saved configuration {chosen['name']} to {ART}/")
    print("  Download pin_f1_xgboost.json and pin_f1_metadata.json into models/")
else:
    for f in ("pin_f1_xgboost.json", "pin_f1_metadata.json"):
        p = os.path.join(ART, f)
        if os.path.exists(p):
            os.rename(p, p + ".rejected")
    print("\n  Existing artefacts renamed to *.rejected — do not deploy them.")
