# 🧪 Hướng dẫn Chạy Thí nghiệm MDEP trên Kaggle

> **Phiên bản:** Commit `2760f4d` (29/06/2026)
> **Tổng số thí nghiệm:** 25 model × 3 seeds = 75 lượt chạy

---

## 📋 Tổng quan Cấu trúc

Thí nghiệm được chia thành **5 script độc lập**, mỗi script chạy trên 1 Kaggle Notebook riêng:

| # | Script | Bộ dữ liệu | Số model | GPU ước tính |
|---|--------|------------|----------|-------------|
| 1 | `run_isic_softmax_baselines.py` | ISIC 2024 | 8 | ~6h (P100) |
| 2 | `run_isic_evidential_baselines.py` | ISIC 2024 | 6 | ~5h |
| 3 | `run_isic_guds_ablations.py` | ISIC 2024 | 11 | ~10h |
| 4 | `run_cifar_suite.py` | CIFAR-100-LT | Tất cả | ~8h |
| 5 | `run_mvtec_suite.py` | MVTec AD | Tất cả | ~3h |

---

## 🔧 Bước 1: Chuẩn bị Kaggle Notebook

### 1.1 Setup Code Cell
Chạy cell Python này ở đầu tiên trong Kaggle Notebook để tự động clone repository và cài đặt thư viện cần thiết. Đoạn code này cũng tự động cập nhật code mới (`git pull`) nếu bạn chạy lại nhiều lần:

```python
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git"
REPO_DIR = Path("/kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning")

def run(cmd, cwd=None):
    cmd = list(map(str, cmd))
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

if REPO_DIR.exists():
    run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

run([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-learn", "matplotlib", "pandas", "h5py", "tqdm", "scipy"
])

os.chdir(REPO_DIR)
print("Repo:", REPO_DIR)
run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR)
```

### 1.3 Add Data (liên kết bộ dữ liệu)
Ở thanh bên phải Notebook → **Add Data**:
- **ISIC:** Tìm `isic-2024-challenge`
- **CIFAR-100:** Không cần — `torchvision` sẽ tự tải
- **MVTec AD:** Tìm `mvtec-ad` hoặc upload bộ chuẩn

### 1.4 Cấu hình phần cứng
- Accelerator: **GPU T4 x2** (Khuyên dùng - code đã được trang bị `TransparentDataParallel` để tự động chạy song song 2 GPU giúp x2 tốc độ) hoặc **GPU P100**.
- Persistence: **Bật** (để không mất output khi session timeout)

---

## 🚀 Bước 2: Chạy Thí nghiệm

### Notebook 1 — Softmax Baselines (ISIC)

Chạy 8 phương pháp long-tailed truyền thống:
`Standard CE` · `Focal Loss` · `Logit Adjustment` · `Class-Balanced CE` · `Balanced Softmax` · `LDAM-DRW` · `cRT` · `MiSLAS`

```python
!python experiments/run_isic_softmax_baselines.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope all \
    --seeds 42 123 456
```

---

### Notebook 2 — Evidential Baselines (ISIC)

Chạy 6 phương pháp Evidential Deep Learning:
`Dense EDL` · `Fisher EDL` · `Flexible EDL` · `R-EDL` · `Static 2:4 EDL` · `RigL-style 2:4`

```python
!python experiments/run_isic_evidential_baselines.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope all \
    --seeds 42 123 456
```

---

### Notebook 3 — GUDS-EDL + Ablations (ISIC)

Chạy mô hình chính và 10 ablation studies:
`full_guds` · `guds_without_pruner` · `guds_without_regrower` · `guds_asymmetric_kl` · `guds_without_efl` · `guds_without_anticryst` · `guds_absolute_pruner` · `guds_class_conditioned_regrower` · `guds_without_topology_cache` · `guds_temperature_only` · `guds_no_posthoc_calibration`

```python
!python experiments/run_isic_guds_ablations.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope all \
    --seeds 42 123 456
```

> **Lưu ý quan trọng:** `full_guds` sử dụng cấu hình chính thức theo bài báo:
> Symmetric KL + KL-Uniform Regrower + Signed First-Order Pruner (Strength=0.5) + EFL (Gamma=5.0).
> Mỗi ablation chỉ thay đổi **đúng 1 biến** so với `full_guds`.

---

### Notebook 4 — CIFAR-100-LT

```python
!python experiments/run_cifar_suite.py \
    --seeds 42 123 456
```

Tham số mặc định: `epochs=100`, `batch_size=128`, `lr=1e-3` (AdamW + CosineAnnealing).
Muốn thay đổi tỷ lệ mất cân bằng:
```python
# Imbalance ratio 1:50
!python experiments/run_cifar_suite.py --ratio 50 --seeds 42 123 456

# Imbalance ratio 1:10
!python experiments/run_cifar_suite.py --ratio 10 --seeds 42 123 456
```

---

### Notebook 5 — MVTec AD

```python
!python experiments/run_mvtec_suite.py \
    --seeds 42 123 456
```

Tham số tự động: `epochs=20`, `batch_size=32` (tự hạ khi nhận diện benchmark=mvtec).
Muốn chạy trên category khác:
```python
!python experiments/run_mvtec_suite.py --category bottle --seeds 42 123 456
```

---

## 📊 Bước 3: Đọc Kết quả

### 3.1 Bảng kết quả trực tiếp trên Console
Mỗi model sau khi train xong sẽ tự động in bảng ASCII được thiết kế riêng cho từng loại dataset:

**Ví dụ (ISIC):**
```text
======================================================================
🏥 CLINICAL EVALUATION (ISIC) | full_guds
======================================================================
Metric                                   |      Value
-----------------------------------------+-----------
 RANKING & DETECTION
  Macro Auroc                            |     0.9234
  Pr Auc                                 |     0.4521
  pAUC (TPR > 0.8) 🌟                    |     0.1876
-----------------------------------------+-----------
 CLINICAL BALANCE
  Balanced Accuracy Default              |     0.8745
...
```

**Ví dụ (MVTec AD):**
```text
======================================================================
🏭 MVTec AD (Anomaly Detection) | full_guds
======================================================================
Metric                                   |      Value
-----------------------------------------+-----------
 ANOMALY DETECTION
  Image Auroc                            |     0.9850
  Image Ap                               |     0.9620
...
```

### 3.2 File output
Mỗi lượt chạy lưu vào `outputs/<ngày>/<tên_model>/`:
- `metrics.json` — Toàn bộ metrics chi tiết
- `run_config.json` — Cấu hình thí nghiệm
- `model_state.pth` — Trọng số model (bỏ qua nếu thêm `--no_save_model`)

Tổng hợp tất cả lượt chạy: `outputs/isic_summary.json`

### 3.3 Tải về máy
```bash
# Zip toàn bộ kết quả
!zip -r /kaggle/working/mdep_results.zip outputs/
```
Sau đó vào tab **Output** ở Kaggle Notebook → Download file zip.

---

## ⚙️ Tham số CLI đầy đủ (ISIC)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 40 | Số epoch huấn luyện |
| `--batch_size` | 32 | Kích thước batch |
| `--lr` | 4e-5 | Learning rate (AdamW) |
| `--seed` | 42 | Random seed đơn |
| `--seeds` | — | Chạy nhiều seeds, VD: `--seeds 42 123 456` |
| `--subsample_ratio` | 20 | Tỷ lệ lấy mẫu con (1/20 dữ liệu ISIC gốc) |
| `--test_ratio` | 0.20 | Tỷ lệ tập test |
| `--no_save_model` | — | Không lưu model weights (tiết kiệm dung lượng) |
| `--no_pretrained` | — | Không dùng ImageNet pretrained |
| `--cpu` | — | Ép chạy trên CPU |
| `--log_every` | 5 | In log mỗi N epoch |
| `--allow_dummy_data` | — | Cho phép dữ liệu giả (chỉ dùng khi dry-run) |

---

## ✅ Đảm bảo Tính Công bằng

Toàn bộ thí nghiệm đã được kiểm chứng (commit `2760f4d`):

1. **Optimizer thống nhất:** Tất cả model dùng AdamW + CosineAnnealing (không có SGD hay fixed LR)
2. **Logit Adjustment:** Đúng công thức Menon et al. ICLR 2021: `logits + log(π_train)`
3. **Không Double-Adjustment:** Các model đã tự bù prior (LA, Balanced Softmax, cRT, MiSLAS) được truyền `p_train = uniform` khi calibrate
4. **Ablation đơn biến:** Mỗi ablation chỉ thay đổi chính xác 1 component so với `full_guds`
5. **Gradient clipping:** `clip_grad_norm_(max_norm=1.0)` cho toàn bộ model
