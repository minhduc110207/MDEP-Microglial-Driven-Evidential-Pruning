# Hướng Dẫn Chi Tiết Chạy Thực Nghiệm MDEP Trên Kaggle

Tài liệu này hướng dẫn chi tiết từng bước thiết lập môi trường, đính kèm dữ liệu, và cung cấp các khối lệnh **Ready-To-Run** ăn liền để chạy toàn bộ các thí nghiệm MDEP trên **Kaggle Notebook**.

---

## ⚡ KHỐI CODE ĂN LIỀN (READY-TO-RUN CHEATSHEET)

Dưới đây là các khối code được thiết kế để bạn chỉ cần **Copy toàn bộ** và **Paste vào 1 Cell duy nhất trên Kaggle** là chạy được ngay lập tức.

### 📦 CELL 1: Cài đặt và Thiết lập Môi trường (Chỉ cần chạy 1 lần duy nhất)
*Tạo 1 Cell mới trên Kaggle, paste toàn bộ phần này vào và chạy:*
```python
# 1. Cài đặt các thư viện bổ trợ cho việc xuất ONNX
!pip install onnx onnxscript

# 2. Xóa thư mục cũ để tránh lỗi trùng lặp khi chạy lại nhiều lần
import shutil
import os
if os.path.exists("MDEP-Microglial-Driven-Evidential-Pruning"):
    shutil.rmtree("MDEP-Microglial-Driven-Evidential-Pruning")

# 3. Clone mã nguồn mới nhất từ GitHub
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git

# 4. Chuyển thư mục hoạt động vào dự án
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

### 🏃 CELL 2: Chạy Thí Nghiệm (Tạo Cell thứ 2, chọn 1 trong các tùy chọn và chạy)

*   **Chạy thuật toán đề xuất chính GUDS-EDL:**
    ```python
    %cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
    !python experiments/isic_paper_experiments.py --experiment full_guds --epochs 15
    ```
*   **Chạy GroupKFold (chia theo nhóm bệnh nhân patient_id):**
    ```python
    %cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
    !python experiments/run_group_kfold.py --folds 5 --epochs 10 --batch_size 32
    ```
*   **Chạy Đa cấu trúc mạng (Backbones):**
    ```python
    %cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
    !python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 10 --batch_size 16
    ```
*   **Chạy so sánh các chế độ Regrowth (Calibration Study):**
    ```python
    %cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
    !python experiments/run_calibration_study.py --modes kl_uniform vacuity ambiguity ratio --epochs 10
    ```
*   **Chạy xuất file ONNX thưa:**
    ```python
    %cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
    !python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
    ```
*   **Chạy kiểm định ngoại quan (Domain Shift & Fairness):**
    ```python
    %cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
    !python experiments/run_external_validation.py
    ```

---

## 🛑 CHI TIẾT TỪNG BƯỚC THAO TÁC TRÊN GIAO DIỆN KAGGLE

### 1️⃣ Bước 1: Tạo Notebook & Cấu Hình Phần Cứng
1. Truy cập [Kaggle](https://www.kaggle.com/) -> Đăng nhập -> Click **"+ New"** ở menu bên trái -> Chọn **"New Notebook"**.
2. **Cấu hình Notebook (Nhìn sang cột cài đặt bên phải):**
   *   Mục **Accelerator**: Click chọn **GPU T4x2** (hoặc GPU T4 tùy phiên bản).
   *   Mục **Internet on**: **Gạt công tắc sang ON** (Kích hoạt Internet để cài thư viện và clone code).

### 2️⃣ Bước 2: Thêm Dữ Liệu ISIC 2024
Vì tập dữ liệu ảnh ISIC 2024 rất nặng ($>10$ GB), bạn cần đính kèm tập dữ liệu có sẵn của Kaggle:
1. Tại cột bên phải của Notebook, click vào nút **"+ Add Input"** (hoặc **"+ Add Data"**).
2. Nhập từ khóa tìm kiếm: `isic-2024-challenge`.
3. Click nút **"Add"** bên cạnh tập dữ liệu chính thức: **"ISIC 2024 - Skin Cancer Detection with 3D-TBP"**.
4. Chờ vài giây để Kaggle đính kèm dữ liệu vào thư mục `/kaggle/input`.

### 3️⃣ Bước 3: Tải Kết Quả Thí Nghiệm Về Máy Cá Nhân
Tất cả kết quả đo đạc (file `.json`, `.csv` và file `.pth` của mô hình) được lưu tự động tại `/kaggle/working/paper_experiment_outputs/`.
1. Nhìn vào tab **"Data"** ở góc bên phải màn hình.
2. Tìm đến mục **"Output"** -> `/kaggle/working`.
3. Click vào biểu tượng ba chấm **`...`** bên cạnh file cần tải (ví dụ: `metrics.json` hoặc file ONNX `dst_edl_resnet18_sparse.onnx`) -> Chọn **"Download"** để lưu về máy cá nhân của bạn.
