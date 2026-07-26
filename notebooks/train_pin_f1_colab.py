# =============================================================================
#  DeepReality — PIN-F1 Meta-Learner Training  (single Colab cell)
# =============================================================================
#  Runtime: T4 or A100.  Set Runtime > Change runtime type > GPU before running.
#
#  BEFORE YOU RUN, upload the trained Layer 2 weights to Google Drive at
#  MyDrive/DeepReality/models/ :
#      pin_b1_clip_ln_tune_final.pt
#      pin_b2_siglip2_finetune_final.pt
#      pin_b3_freq_cnn_final.pt
#      blaze_face_short_range.tflite
#      pin_b4_ai_deepfake_real/{config.json,preprocessor_config.json,model.safetensors}
#
#  Then set REPO_URL below to your GitHub repository.
#
#  Output, written to MyDrive/DeepReality/artifacts/ :
#      pin_f1_xgboost.json     -> copy into your local models/
#      pin_f1_metadata.json    -> copy into your local models/
#      f1_features.parquet     -> the extracted design matrix, for reuse
# =============================================================================

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
REPO_URL      = "https://github.com/OmerKurtulus/DeepReality.git"  # <-- EDIT
REPO_BRANCH   = "main"

RUN_TAG       = "v2"   # checkpoints are namespaced by this; bump it to start clean

N_TRAIN       = 800    # balanced samples from the training corpus
N_HOLDOUT     = 300    # balanced samples from a DIFFERENT corpus
RESUME        = True   # continue from a checkpoint carrying the same RUN_TAG

PROFILE_FIRST = 3      # print per-pin timings for the first N images
MAX_WORKERS   = 4      # concurrent pins; see the note in section 1
TORCH_THREADS = 2      # intra-op threads per PyTorch call

TRAIN_DATASET = "Hemg/deepfake-and-real-images"      # 0=Fake, 1=Real
HOLDOUT_DATASET = "ComplexDataLab/OpenFake"
HOLDOUT_CONFIG  = "core"                             # this corpus requires a config

DRIVE_ROOT    = "/content/drive/MyDrive/DeepReality"

# =============================================================================

import os, sys, io, json, time, math, shutil, subprocess, warnings
warnings.filterwarnings("ignore")

def hdr(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}", flush=True)

# ----------------------------------------------------------------------------
hdr("1 · ENVIRONMENT")
# ----------------------------------------------------------------------------
import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"  Device : {GPU}")
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM   : {total:.1f} GB")
    torch.backends.cudnn.benchmark = True
    # A100 supports TF32 matmul, which is a free ~1.5x on the ViT forwards
    if "A100" in GPU or "H100" in GPU:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  TF32   : enabled")
else:
    print("  WARNING: no GPU. Feature extraction will be very slow.")

# Thread budget. The orchestrator runs several pins concurrently and each
# PyTorch call spawns its own intra-op pool; left at their defaults these
# multiply into far more OS threads than there are cores, and the
# resulting contention costs more than the concurrency gains. The image
# processors in particular do their resizing and normalisation on the CPU.
import multiprocessing
torch.set_num_threads(TORCH_THREADS)
os.environ["OMP_NUM_THREADS"] = str(TORCH_THREADS)
print(f"  vCPU   : {multiprocessing.cpu_count()}   "
      f"torch threads: {TORCH_THREADS}   pin workers: {MAX_WORKERS}")

# ----------------------------------------------------------------------------
hdr("2 · DRIVE + REPOSITORY")
# ----------------------------------------------------------------------------
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

ART_DIR = os.path.join(DRIVE_ROOT, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

REPO_DIR = "/content/DeepReality"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH,
                    REPO_URL, REPO_DIR], check=True)
print(f"  Repository : {REPO_DIR}")

# Link the fine-tuned weights from Drive rather than copying several GB
drive_models = os.path.join(DRIVE_ROOT, "models")
repo_models  = os.path.join(REPO_DIR, "models")
assert os.path.isdir(drive_models), f"Upload the weights to {drive_models} first"

REQUIRED_WEIGHTS = (
    "pin_b1_clip_ln_tune_final.pt",
    "pin_b2_siglip2_finetune_final.pt",
    "pin_b3_freq_cnn_final.pt",
)
missing = [w for w in REQUIRED_WEIGHTS if not os.path.exists(os.path.join(drive_models, w))]
if missing:
    raise FileNotFoundError(
        f"Missing from {drive_models}: {missing}\n"
        f"Present: {sorted(os.listdir(drive_models))}"
    )

for name in REQUIRED_WEIGHTS:
    dst = os.path.join(repo_models, name)
    if not os.path.exists(dst):
        os.symlink(os.path.join(drive_models, name), dst)

# PIN-B4 is a public pretrained model, so its weights are fetched directly
# rather than requiring a manual upload. The two small config files ship
# with the repository; only the tensor file is absent.
b4_dir = os.path.join(repo_models, "pin_b4_ai_deepfake_real")
os.makedirs(b4_dir, exist_ok=True)
b4_weights = os.path.join(b4_dir, "model.safetensors")
if not os.path.exists(b4_weights):
    drive_b4 = os.path.join(drive_models, "pin_b4_ai_deepfake_real", "model.safetensors")
    if os.path.exists(drive_b4):
        os.symlink(drive_b4, b4_weights)
        print("  PIN-B4     : linked from Drive")
    else:
        print("  PIN-B4     : downloading from Hugging Face (~354 MB)...")
        from huggingface_hub import hf_hub_download
        for fname in ("config.json", "preprocessor_config.json", "model.safetensors"):
            target = os.path.join(b4_dir, fname)
            if not os.path.exists(target):
                shutil.copy(
                    hf_hub_download("prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2", fname),
                    target,
                )

# PIN-A4's face detector ships with the repository, but fetch it if absent
blaze = os.path.join(repo_models, "blaze_face_short_range.tflite")
if not os.path.exists(blaze):
    print("  PIN-A4     : downloading BlazeFace...")
    subprocess.run([
        "wget", "-q", "-O", blaze,
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ], check=True)

print(f"  Weights    : {sorted(os.listdir(repo_models))}")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ----------------------------------------------------------------------------
hdr("3 · DEPENDENCIES")
# ----------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "transformers>=4.40.0", "c2pa-python>=0.28.0", "mediapipe>=0.10.0",
                "pillow-heif>=0.16.0", "PyWavelets>=1.5.0", "xgboost>=2.0.0",
                "datasets>=2.19.0", "scikit-learn>=1.4.0", "pyarrow>=15.0.0"],
               check=False)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
print("  Installed.")

# ----------------------------------------------------------------------------
hdr("4 · DATASETS")
# ----------------------------------------------------------------------------
from datasets import load_dataset, Image as HFImage

def take_balanced(name, n, label_col, label_map, split="train", cfg=None):
    """
    Stream a balanced sample without downloading the whole corpus.

    decode=False keeps the ORIGINAL encoded bytes. This matters: PIN-A1,
    A2 and A3 read the container itself, so re-encoding through PIL would
    destroy the compression history that Error Level Analysis depends on
    and would fabricate a uniform format across the corpus.
    """
    ds = load_dataset(name, cfg, split=split, streaming=True) if cfg else \
         load_dataset(name, split=split, streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))
    per_class, out = n // 2, {0: [], 1: []}
    for row in ds:
        y = label_map(row[label_col])
        if y is None or len(out[y]) >= per_class:
            if len(out[0]) >= per_class and len(out[1]) >= per_class:
                break
            continue
        img = row["image"]
        data = img["bytes"] if isinstance(img, dict) else None
        if not data:
            continue
        out[y].append(data)
    rows = [(b, 0) for b in out[0]] + [(b, 1) for b in out[1]]
    print(f"  {name}: {len(out[0])} authentic + {len(out[1])} synthetic")
    return rows

# Canonical label convention for THIS project: 1 = synthetic, 0 = authentic.
# The training corpus uses 0=Fake, 1=Real, so it is inverted here.
train_rows = take_balanced(
    TRAIN_DATASET, N_TRAIN, "label",
    lambda v: 1 if int(v) == 0 else 0)

try:
    holdout_rows = take_balanced(
        HOLDOUT_DATASET, N_HOLDOUT, "label",
        lambda v: 1 if str(v).lower() in ("fake", "1", "true") else 0,
        split="test", cfg=HOLDOUT_CONFIG)
except Exception as e:
    print(f"  Holdout corpus unavailable ({type(e).__name__}: {e})")
    print("  Continuing without a cross-dataset holdout; the calibration")
    print("  split still provides an in-distribution estimate.")
    holdout_rows = []

# ----------------------------------------------------------------------------
hdr("5 · FEATURE EXTRACTION")
# ----------------------------------------------------------------------------
from core.pipeline import PinPipeline
from layer1_preprocessing.pin_a1_metadata import PinA1Metadata
from layer1_preprocessing.pin_a2_c2pa import PinA2C2pa
from layer1_preprocessing.pin_a3_ela import PinA3Ela
from layer1_preprocessing.pin_a4_face import PinA4Face
from layer2_detection_core.pin_b1_clip import PinB1Clip
from layer2_detection_core.pin_b2_siglip2 import PinB2Siglip
from layer2_detection_core.pin_b3_freq import PinB3Freq
from layer2_detection_core.pin_b4_IndependentCore import PinB4IndependentCore
from layer4_xai.pin_d1_gradcam import PinD1GradCam
from layer4_xai.pin_d2_anomaly import PinD2AnomalyLocalization
from layer6_ensemble.feature_extractor import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features)

# transformers resolves submodules lazily and that first resolution is not
# thread-safe; warm every symbol before the concurrent stage starts.
from transformers import (CLIPModel, CLIPProcessor, AutoModel, AutoProcessor,
                          AutoImageProcessor, SiglipForImageClassification)

def build_feature_pipeline():
    """Layers 1, 2 and 4 only — E1 needs an API key, F1 is what we are fitting."""
    p = PinPipeline(max_workers=MAX_WORKERS)
    for pin in (PinA1Metadata(), PinA2C2pa(), PinA3Ela(), PinA4Face(),
                PinB1Clip(), PinB2Siglip(), PinB3Freq(), PinB4IndependentCore()):
        p.add_pin(pin)
    p.add_pin(PinD1GradCam(), depends_on=["PIN-B1", "PIN-B2", "PIN-B3"])
    p.add_pin(PinD2AnomalyLocalization(), depends_on=["PIN-A3", "PIN-B3", "PIN-D1"])
    return p

import pandas as pd
WORK = "/content/_work"; os.makedirs(WORK, exist_ok=True)

def extract(rows, tag):
    """
    Run the pipeline over a corpus and assemble the design matrix.

    Checkpoints are namespaced by RUN_TAG so a re-run never collides with
    the artefacts of an earlier one, and resume is by row count: the
    streaming order is deterministic for a fixed dataset, so skipping the
    first N rows reproduces exactly the samples already processed.
    """
    ckpt = os.path.join(ART_DIR, f"f1_features_{RUN_TAG}_{tag}.parquet")

    records, start = [], 0
    if RESUME and os.path.exists(ckpt):
        prior = pd.read_parquet(ckpt)
        if len(prior) >= len(rows):
            print(f"  [{tag}] checkpoint complete: {len(prior)} rows — reused")
            return prior
        records = prior.to_dict("records")
        start = len(records)
        print(f"  [{tag}] resuming from checkpoint at {start}/{len(rows)}")

    pipeline = build_feature_pipeline()
    t0 = time.time()

    for i in range(start, len(rows)):
        raw, y = rows[i]
        n = i + 1
        path = os.path.join(WORK, f"{tag}_{n:06d}")
        try:
            with open(path, "wb") as fh:
                fh.write(raw)

            if n - start <= PROFILE_FIRST:
                # Instrument the opening images so a throughput problem is
                # attributed to a specific pin within seconds rather than
                # inferred from the aggregate rate an hour later.
                timings = {}
                run = pipeline.run(
                    path,
                    on_pin_complete=lambda pid, res, dt: timings.__setitem__(pid, dt),
                )
                slowest = sorted(timings.items(), key=lambda kv: -kv[1])
                print(f"  [profile {n}] total {run.total_time:.2f}s  |  " +
                      "  ".join(f"{p}:{d:.2f}s" for p, d in slowest[:6]), flush=True)
            else:
                run = pipeline.run(path)

            feats = extract_features(run.results)
            feats["__label__"] = y
            records.append(feats)
        except Exception as exc:
            print(f"    skip {n}: {type(exc).__name__}: {exc}")
        finally:
            if os.path.exists(path):
                os.remove(path)

        done = n - start
        if done % 25 == 0 or n == len(rows):
            el = time.time() - t0
            rate = el / max(done, 1)
            print(f"  [{tag}] {n}/{len(rows)}  {rate:.2f} s/img  "
                  f"ETA {rate * (len(rows) - n) / 60:.1f} min", flush=True)
        if done % 100 == 0:                   # survive a disconnect
            pd.DataFrame(records).to_parquet(ckpt, index=False)

    df = pd.DataFrame(records)
    df.to_parquet(ckpt, index=False)
    print(f"  [{tag}] done: {df.shape}  ->  {ckpt}")
    return df

train_df = extract(train_rows, "train")
holdout_df = extract(holdout_rows, "holdout") if holdout_rows else None

# ----------------------------------------------------------------------------
hdr("6 · TRAINING")
# ----------------------------------------------------------------------------
import numpy as np, xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             brier_score_loss, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from config.settings import ENSEMBLE_CONFIG

TP = ENSEMBLE_CONFIG["training"]
X = train_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
y = train_df["__label__"].to_numpy(dtype=np.int32)
print(f"  Design matrix : {X.shape}   positives: {y.sum()}/{len(y)}")

const = [FEATURE_NAMES[i] for i in range(X.shape[1])
         if np.nanstd(X[:, i]) < 1e-9]
if const:
    print(f"  Zero-variance in this corpus ({len(const)}): {const}")
    print("  These are provenance features stripped during dataset packaging.")
    print("  They are retained in the schema so a corpus that preserves")
    print("  metadata can activate them on retraining without a version bump.")

# Split first, then calibrate on data the booster never saw. Calibrating on
# the training split is the classic error: it produces a sigmoid fitted to
# memorised scores and reports excellent, meaningless calibration.
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=TP["random_state"])

params = dict(
    objective="binary:logistic", eval_metric="auc", tree_method="hist",
    max_depth=TP["max_depth"], learning_rate=TP["learning_rate"],
    subsample=TP["subsample"], colsample_bytree=TP["colsample_bytree"],
    min_child_weight=TP["min_child_weight"], reg_lambda=TP["reg_lambda"],
    random_state=TP["random_state"],
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Cross-validated estimate before fitting the deliverable model
skf = StratifiedKFold(n_splits=TP["cv_folds"], shuffle=True,
                      random_state=TP["random_state"])
cv_auc = []
for tr, va in skf.split(X_fit, y_fit):
    m = xgb.XGBClassifier(n_estimators=TP["n_estimators"],
                          early_stopping_rounds=TP["early_stopping_rounds"],
                          **params)
    m.fit(X_fit[tr], y_fit[tr], eval_set=[(X_fit[va], y_fit[va])], verbose=False)
    cv_auc.append(roc_auc_score(y_fit[va], m.predict_proba(X_fit[va])[:, 1]))
print(f"  {TP['cv_folds']}-fold CV ROC-AUC : "
      f"{np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

model = xgb.XGBClassifier(n_estimators=TP["n_estimators"],
                          early_stopping_rounds=TP["early_stopping_rounds"],
                          **params)
model.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], verbose=False)
print(f"  Boosting rounds used : {model.best_iteration + 1}")

# ----------------------------------------------------------------------------
hdr("7 · CALIBRATION (Platt)")
# ----------------------------------------------------------------------------
raw_cal = model.predict_proba(X_cal)[:, 1]
platt = LogisticRegression(C=1e6, solver="lbfgs")
platt.fit(raw_cal.reshape(-1, 1), y_cal)
A, B = float(platt.coef_[0][0]), float(platt.intercept_[0])
calibrate = lambda r: 1.0 / (1.0 + np.exp(-np.clip(A * r + B, -60, 60)))
print(f"  a = {A:.4f}   b = {B:.4f}")
print(f"  Brier  raw {brier_score_loss(y_cal, raw_cal):.4f}"
      f"  ->  calibrated {brier_score_loss(y_cal, calibrate(raw_cal)):.4f}")

def ece(y_true, p, bins=10):
    """Expected Calibration Error: mean |confidence - accuracy| per bin."""
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(total)

def report(name, y_true, p):
    pred = (p >= ENSEMBLE_CONFIG["decision_threshold"]).astype(int)
    met = dict(roc_auc=round(float(roc_auc_score(y_true, p)), 4),
               accuracy=round(float(accuracy_score(y_true, pred)), 4),
               f1=round(float(f1_score(y_true, pred)), 4),
               brier=round(float(brier_score_loss(y_true, p)), 4),
               ece=round(ece(y_true, p), 4), n=int(len(y_true)))
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"\n  {name}  (n={met['n']})")
    print(f"    ROC-AUC {met['roc_auc']}  acc {met['accuracy']}  "
          f"F1 {met['f1']}  Brier {met['brier']}  ECE {met['ece']}")
    print(f"    TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    return met

hdr("8 · EVALUATION")
cal_metrics = report("Calibration split (in-distribution)", y_cal, calibrate(raw_cal))

holdout_metrics = None
if holdout_df is not None and len(holdout_df):
    Xh = holdout_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    yh = holdout_df["__label__"].to_numpy(dtype=np.int32)
    holdout_metrics = report(
        f"CROSS-DATASET holdout — {HOLDOUT_DATASET}", yh,
        calibrate(model.predict_proba(Xh)[:, 1]))
    print("\n  Cross-dataset performance is the figure that matters: it is the")
    print("  only one measured on a generator distribution absent from both")
    print("  the base detectors' training and this meta-learner's.")

# Best single detector, as the bar the ensemble must clear
print("\n  Baseline comparison (calibration split):")
for feat in ("b1_prob", "b2_prob", "b3_prob", "b4_fake_score", "consensus_mean"):
    idx = FEATURE_NAMES.index(feat)
    col = X_cal[:, idx]
    ok = ~np.isnan(col)
    if ok.sum() > 10 and len(np.unique(y_cal[ok])) > 1:
        print(f"    {feat:<18} ROC-AUC {roc_auc_score(y_cal[ok], col[ok]):.4f}")
print(f"    {'PIN-F1 (ensemble)':<18} ROC-AUC {cal_metrics['roc_auc']:.4f}")

# ----------------------------------------------------------------------------
hdr("9 · FEATURE IMPORTANCE")
# ----------------------------------------------------------------------------
booster = model.get_booster()
booster.feature_names = list(FEATURE_NAMES)
gain = booster.get_score(importance_type="gain")
importance = {n: float(gain.get(n, 0.0)) for n in FEATURE_NAMES}
for n, g in sorted(importance.items(), key=lambda kv: -kv[1])[:15]:
    if g > 0:
        print(f"    {n:<34} {g:10.2f}")

# ----------------------------------------------------------------------------
hdr("10 · SAVE ARTEFACTS")
# ----------------------------------------------------------------------------
model_out = os.path.join(ART_DIR, "pin_f1_xgboost.json")
meta_out  = os.path.join(ART_DIR, "pin_f1_metadata.json")
booster.save_model(model_out)

metadata = {
    "feature_schema_version": FEATURE_SCHEMA_VERSION,
    "feature_names": list(FEATURE_NAMES),
    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "training_dataset": TRAIN_DATASET,
    "holdout_dataset": HOLDOUT_DATASET if holdout_metrics else None,
    "training_samples": int(len(X_fit)),
    "calibration_samples": int(len(X_cal)),
    "label_convention": "1 = synthetic, 0 = authentic",
    "calibration": {"method": "platt", "a": A, "b": B},
    "cv_roc_auc_mean": round(float(np.mean(cv_auc)), 4),
    "cv_roc_auc_std": round(float(np.std(cv_auc)), 4),
    "calibration_split_metrics": cal_metrics,
    "holdout_metrics": holdout_metrics or cal_metrics,
    "feature_importance": importance,
    "zero_variance_features": const,
    "hyperparameters": {k: v for k, v in TP.items()},
    "best_iteration": int(model.best_iteration),
}
with open(meta_out, "w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2, ensure_ascii=False)

print(f"  {model_out}")
print(f"  {meta_out}")
print("""
  NEXT STEPS
  ----------
  1. Download both files from Drive: MyDrive/DeepReality/artifacts/
  2. Place them in your local models/ directory
  3. Run  python3 main.py  — PIN-F1 will report model_status: "trained"

  The feature checkpoint (f1_features_train.parquet) is kept in Drive, so a
  retrain with different hyperparameters costs seconds rather than hours.
""")
