# Hướng Dẫn Chi Tiết Chạy Thực Nghiệm MDEP Trên Kaggle

Tài liệu này tích hợp các bước thiết lập tối ưu từ `kaggle_instructions.md` và cung cấp mã nguồn **Ready-to-Run** chuẩn nhất để bạn chạy toàn bộ các thí nghiệm MDEP (cả **ISIC 2024** và **CIFAR-100-LT**) trên **Kaggle Notebook**.

---

## ⚡ KHỐI CODE ĂN LIỀN (READY-TO-RUN CHEATSHEET)

Dưới đây là các khối code được thiết kế để bạn chỉ cần **Copy toàn bộ** và **Paste vào 1 Cell duy nhất trên Kaggle** là chạy được ngay lập tức.

### 📦 CELL 1: Cài đặt, Cấu hình Môi trường & Liên kết Dữ liệu (Chỉ cần chạy 1 lần duy nhất)
*Tạo 1 Cell mới trên Kaggle, paste toàn bộ đoạn code Python dưới đây vào và bấm Run. Lệnh này sẽ tự động tải thư viện, kéo mã nguồn mới nhất từ GitHub, cấu hình tăng tốc CUDA và liên kết dữ liệu ISIC/CIFAR-100:*

```python
import os
import subprocess
import sys
import shutil
from pathlib import Path

REPO_URL = "https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git"
REPO_DIR = Path("/kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning")

def run(cmd, cwd=None):
    cmd = list(map(str, cmd))
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

# 1. Đồng bộ mã nguồn từ GitHub (Pull nếu đã tồn tại, Clone nếu chưa có)
if REPO_DIR.exists():
    try:
        run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
        print("Mã nguồn đã được cập nhật thành công (git pull).")
    except Exception:
        print("Lỗi git pull, đang xóa thư mục cũ để clone lại từ đầu...")
        shutil.rmtree(REPO_DIR)
        run(["git", "clone", REPO_URL, str(REPO_DIR)])
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

# 2. Cài đặt các thư viện cần thiết và thư viện xuất ONNX
run([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-learn", "matplotlib", "pandas", "h5py", "tqdm", "scipy", "onnx", "onnxscript"
])

# 3. Cấu hình các biến môi trường tối ưu hóa phần cứng và W&B
os.environ.setdefault("MDEP_NUM_WORKERS", "4")
os.environ.setdefault("MDEP_PREFETCH_FACTOR", "4")
os.environ.setdefault("MDEP_CUDNN_BENCHMARK", "1")
os.environ.setdefault("MDEP_MATMUL_PRECISION", "high")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# 4. Tự động phát hiện và liên kết tập dữ liệu ISIC 2024
def find_isic_root():
    candidates = [
        Path("/kaggle/input/isic-2024-challenge"),
        Path("/kaggle/input"),
        REPO_DIR / "data" / "isic-2024-challenge",
    ]
    for base in candidates:
        if not base.exists():
            continue
        if (base / "train-metadata.csv").exists():
            return base
        for path in base.rglob("train-metadata.csv"):
            return path.parent
    return None

isic_root = find_isic_root()
if isic_root is None:
    isic_root = REPO_DIR / "data" / "isic-2024-challenge"
    isic_root.mkdir(parents=True, exist_ok=True)
    # Tải qua Kaggle API nếu không tìm thấy dữ liệu mount sẵn
    try:
        run(["kaggle", "competitions", "download", "-c", "isic-2024-challenge", "-p", isic_root])
        for archive in isic_root.glob("*.zip"):
            run(["unzip", "-q", "-n", archive, "-d", isic_root])
    except Exception as e:
        print(f"Cảnh báo: Không thể tải tập dữ liệu tự động ({e}). Hãy đính kèm dữ liệu thủ công.")

os.environ["ISIC_ROOT"] = str(isic_root)

# 5. Liên kết tập dữ liệu CIFAR-100 Python nếu có
def link_cifar_dataset():
    candidates = [
        Path("/kaggle/input/cifar-100-python"),
        Path("/kaggle/input/cifar100"),
        Path("/kaggle/input/cifar-100"),
    ]
    cifar_source = None
    for base in candidates:
        if (base / "train").exists() and (base / "meta").exists():
            cifar_source = base
            break
    if cifar_source is not None:
        cifar_target_dir = REPO_DIR / "data"
        cifar_target = cifar_target_dir / "cifar-100-python"
        cifar_target_dir.mkdir(parents=True, exist_ok=True)
        if not cifar_target.exists():
            try:
                os.symlink(cifar_source, cifar_target)
                print("Đã liên kết CIFAR-100 từ:", cifar_source)
            except OSError as e:
                print("Cảnh báo symlink CIFAR-100:", e)

link_cifar_dataset()

# 6. Chuyển thư mục hoạt động vào dự án
os.chdir(REPO_DIR)
print("\n" + "="*50)
print("THIẾT LẬP THÀNH CÔNG (SETUP COMPLETE).")
print("Thư mục chạy:", REPO_DIR)
print("Đường dẫn ISIC_ROOT:", os.environ["ISIC_ROOT"])
print("="*50, flush=True)
```

---

### 🏃 CELL 2: Chạy Thí Nghiệm (Tạo Cell thứ 2, chọn 1 trong các ô dưới đây và chạy)

Sau khi Cell 1 in ra dòng `THIẾT LẬP THÀNH CÔNG`, hãy chọn 1 trong các lệnh dưới đây để dán vào Cell 2 và chạy:

#### 🔹 Thí nghiệm 2.1: Chạy mô hình đề xuất chính GUDS-EDL (ISIC)
```python
# Chạy mô hình đề xuất chính
!python experiments/isic_paper_experiments.py --experiment full_guds --epochs 40 --batch_size 32 --seeds 42 --split_seed 42 --no_save_model
```

#### 🔹 Thí nghiệm 2.2: Chạy GroupKFold chéo theo nhóm bệnh nhân (ISIC)
```python
# Chạy GroupKFold 5-Fold
!python experiments/run_group_kfold.py --folds 5 --epochs 15 --batch_size 32
```

#### 🔹 Thí nghiệm 2.3: Chạy đa kiến trúc mạng Backbones (ISIC)
```python
# Chạy đánh giá mở rộng trên ResNet-18, ConvNeXt-T, Swin-T
!python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 15 --batch_size 16
```

#### 🔹 Thí nghiệm 2.4: Chạy Nghiên cứu Calibration & Các Regrowth Ablations (ISIC)
```python
# Chạy so sánh hiệu chuẩn của các chế độ Regrowth (KL, Vacuity, Ambiguity, Ratio)
!python experiments/run_calibration_study.py --modes kl_uniform vacuity ambiguity ratio --epochs 15 --batch_size 32
```

#### 🔹 Thí nghiệm 2.5: Chạy xuất file ONNX thưa và đo đạc phần cứng
```python
# Chạy xuất file ONNX thưa
!python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
```

#### 🔹 Thí nghiệm 2.6: Chạy kiểm định ngoại quan (Domain Shift & Fairness)
```python
# Chạy đánh giá ngoại quan trên các loại da Fitzpatrick
!python experiments/run_external_validation.py
```

#### 🔹 Thí nghiệm 2.7: Chạy thực nghiệm trên tập dữ liệu CIFAR-100-LT (Long-tail)
```python
# Chạy thực nghiệm CIFAR-100-LT với tỉ lệ mất cân bằng 100
!python -u experiments/run_cifar_suite.py --ratio 100 --experiment full_guds --experiment standard_ce --experiment dense_edl --epochs 100 --batch_size 128 --seeds 42
```

---

## 🛑 THAO TÁC CƠ BẢN TRÊN GIAO DIỆN KAGGLE

### 1️⃣ Bước 1: Khởi tạo Notebook & Bật GPU
1. Vào Kaggle -> Đăng nhập -> Click **"+ New"** -> Chọn **"New Notebook"**.
2. Tại cột menu bên phải màn hình:
   *   Mục **Accelerator**: Click chọn **GPU T4x2** (hoặc GPU T4).
   *   Mục **Internet on**: **Gạt công tắc sang ON** (cực kỳ quan trọng để cài thư viện).

### 2️⃣ Bước 2: Đính kèm dữ liệu (Add Dataset)
1. Tại cột bên phải Notebook, click nút **"+ Add Input"** (hoặc **"+ Add Data"**).
2. Tìm kiếm: `isic-2024-challenge` -> Click **"Add"** cạnh dữ liệu cuộc thi chính thức.
3. Tìm kiếm: `cifar-100-python` -> Click **"Add"** để liên kết nếu muốn chạy thí nghiệm CIFAR-100-LT.

### 3️⃣ Bước 3: Tải Kết Quả Thí Nghiệm
Kết quả đo đạc được lưu tự động tại `/kaggle/working/paper_experiment_outputs/`.
1. Nhìn sang tab **"Data"** ở góc trên bên phải màn hình.
2. Tìm đến mục **"Output"** -> `/kaggle/working`.
3. Click biểu tượng ba chấm **`...`** bên cạnh file cần tải -> Chọn **"Download"** (ví dụ: các đồ thị `.png` hoặc file `.json`).
