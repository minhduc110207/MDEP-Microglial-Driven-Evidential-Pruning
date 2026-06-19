"""
============================================================================
  Config — All hyperparameters and paths in one place.
  
  Modify this file to change experiment settings without touching other code.
============================================================================
"""

import os
import torch


# ── Paths ──────────────────────────────────────────────────────────────
DATA_SEARCH_DIRS = [
    r"E:\Testing\mdep\isic-2024-challenge",  # Local path
    "/kaggle/input",                          # Kaggle root
    "/kaggle/input/competitions",             # Kaggle competitions
]

OUTPUT_DIR = "/kaggle/working/" if os.path.exists("/kaggle/working/") else "./"

# ── Device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ───────────────────────────────────────────────────────────────
NUM_CLASSES      = 2
IMAGE_SIZE       = 224
BATCH_SIZE       = 32
TEST_RATIO       = 0.2
SUBSAMPLE_RATIO  = 20      # benign-to-malignant ratio cap (None = no cap)
NUM_WORKERS      = 4

# ── Model ──────────────────────────────────────────────────────────────
BACKBONE         = "resnet18"
PRETRAINED       = True
FREEZE_BACKBONE  = True     # Core of F-EDL: only train the head

# ── F-EDL Head ─────────────────────────────────────────────────────────
EVIDENCE_ACTIVATION = "softplus"    # activation for α and τ heads
ALPHA_OFFSET        = 1.0           # α = softplus(logit) + offset  (ensures α > 1)
TAU_OFFSET          = 1e-6          # τ = softplus(logit) + offset  (ensures τ > 0)
HEAD_INIT_STD       = 0.01          # small init to avoid KL explosion

# ── Loss ───────────────────────────────────────────────────────────────
REG_LAMBDA          = 0.1           # weight for Brier-score regularization
DISP_REG_WEIGHT     = 0.01         # weight for dispersion regularization
ANNEALING_EPOCHS    = 10            # epochs to linearly ramp up regularization

# ── Optimizer ──────────────────────────────────────────────────────────
LEARNING_RATE    = 1e-3             # higher LR since only head is trained
WEIGHT_DECAY     = 1e-4
GRAD_CLIP_NORM   = 1.0

# ── Scheduler ──────────────────────────────────────────────────────────
SCHEDULER        = "cosine"         # "cosine" | "step" | "none"
COSINE_T_MAX     = None             # defaults to TOTAL_EPOCHS if None

# ── Training ───────────────────────────────────────────────────────────
TOTAL_EPOCHS     = 40
PRINT_INTERVAL   = 200              # batches between progress prints
KAGGLE_TIME_LIMIT = 29500           # seconds (≈8.2 hours safety margin)

# ── Checkpointing ─────────────────────────────────────────────────────
CHECKPOINT_LATEST = os.path.join(OUTPUT_DIR, "latest_checkpoint.pth")
CHECKPOINT_BEST   = os.path.join(OUTPUT_DIR, "best_checkpoint.pth")
MODEL_SAVE_PATH   = os.path.join(OUTPUT_DIR, "fedl_model.pth")

# ── WandB ──────────────────────────────────────────────────────────────
WANDB_PROJECT    = "F-EDL-ResNet18"
WANDB_RUN_NAME   = "F-EDL-Main-Run"

# ── Evaluation ─────────────────────────────────────────────────────────
ECE_BINS         = 15
PAUC_MIN_TPR     = 0.80
