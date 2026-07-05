# Hướng Dẫn Từng Bước Chạy Thực Nghiệm MDEP Trên Kaggle

Tài liệu này hướng dẫn chi tiết từng bước (Step-by-step) dành cho người mới để thiết lập môi trường và chạy toàn bộ các thí nghiệm MDEP trên **Kaggle Notebook**.

---

## 🛑 BƯỚC 1: Tạo Kaggle Notebook & Bật Cấu Hình Phần Cứng

1. Truy cập vào [Kaggle](https://www.kaggle.com/) và đăng nhập tài khoản của bạn.
2. Click vào nút **"+ New"** ở menu bên trái -> Chọn **"New Notebook"**.
3. **Cấu hình Notebook (Cực kỳ quan trọng - Nhìn sang cột cài đặt bên phải):**
   *   Mục **Accelerator**: Click chọn **GPU T4x2** (hoặc GPU T4 tùy phiên bản).
   *   Mục **Internet on**: **Gạt công tắc sang ON** (Kích hoạt Internet để clone code từ GitHub và cài thư viện).

---

## 📂 BƯỚC 2: Thêm Dữ Liệu ISIC 2024 Vào Notebook

Vì tập dữ liệu ảnh da liễu ISIC 2024 rất nặng ($>10$ GB), bạn cần đính kèm tập dữ liệu có sẵn của Kaggle vào Notebook thay vì tự upload:

1. Tại cột bên phải của Notebook, click vào nút **"+ Add Input"** (hoặc **"+ Add Data"**).
2. Nhập từ khóa tìm kiếm: `isic-2024-challenge`.
3. Click nút **"Add"** bên cạnh tập dữ liệu chính thức của cuộc thi: **"ISIC 2024 - Skin Cancer Detection with 3D-TBP"**.
4. Chờ vài giây để Kaggle đính kèm dữ liệu vào thư mục `/kaggle/input`.

---

## ⚙️ BƯỚC 3: Cài Đặt Thư Viện & Clone Code (Chạy trong Cell 1)

Tạo một Cell code mới, dán đoạn lệnh sau vào và bấm **Run (nút Play hoặc Ctrl+Enter)**:

```python
# 1. Cài đặt các thư viện bổ trợ cho việc xuất ONNX
!pip install onnx onnxscript

# 2. Clone mã nguồn từ Github của bạn về thư mục làm việc của Kaggle
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git

# 3. Di chuyển thư mục làm việc vào trong repo vừa tải
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

---

## 🏃 BƯỚC 4: Chạy Các Thực Nghiệm (Chạy trong Cell 2)

Sau khi Cell 1 chạy xong và báo thành công, hãy tạo tiếp một Cell code mới. Bạn có thể copy bất kỳ câu lệnh nào dưới đây để chạy thí nghiệm tương ứng:

### 🔹 Phương án 4.1: Chạy Thí Nghiệm Chính (ResNet-18 thưa trên ISIC 2024)
Chạy thuật toán đề xuất GUDS-EDL:
```python
!python experiments/isic_paper_experiments.py --experiment full_guds --epochs 15
```

### 🔹 Phương án 4.2: Chạy 5-Fold GroupKFold theo nhóm bệnh nhân
```python
!python experiments/run_group_kfold.py --folds 5 --epochs 10 --batch_size 32
```

### 🔹 Phương án 4.3: Chạy Đánh giá Đa cấu trúc (Backbones)
```python
!python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 10 --batch_size 16
```

### 🔹 Phương án 4.4: Chạy Nghiên cứu Calibration & Các regrowth ablations
```python
!python experiments/run_calibration_study.py --modes kl_uniform vacuity ambiguity ratio --epochs 10
```

### 🔹 Phương án 4.5: Chạy xuất file ONNX thưa
```python
!python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
```

---

## 📥 BƯỚC 5: Tải Kết Quả Thí Nghiệm Về Máy Cá Nhân

Tất cả kết quả đo đạc (file `.json`, `.csv` và checkpoint `.pth` của mô hình) được lưu tự động tại thư mục `/kaggle/working/paper_experiment_outputs/`.

Để tải các kết quả này về máy của bạn:
1. Nhìn vào tab **"Data"** ở góc bên phải màn hình.
2. Tìm đến mục **"Output"** -> `/kaggle/working`.
3. Bạn sẽ thấy thư mục `paper_experiment_outputs` xuất hiện tại đây sau khi chạy xong thí nghiệm.
4. Click vào biểu tượng ba chấm **`...`** bên cạnh file cần tải (ví dụ: `metrics.json` hoặc file ONNX `dst_edl_resnet18_sparse.onnx`) -> Chọn **"Download"** để lưu về máy cá nhân của bạn.
