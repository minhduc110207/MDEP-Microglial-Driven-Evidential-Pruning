# Kaggle Ready-To-Run Cheatsheet

Dưới đây là các khối code được thiết kế để bạn chỉ cần **Copy toàn bộ** và **Paste vào 1 Cell duy nhất trên Kaggle** là có thể chạy ngay lập tức mà không cần gõ thêm bất kỳ lệnh nào khác.

---

## 🚀 CELL 1: Cài đặt và Thiết lập Môi trường (Chỉ cần chạy 1 lần duy nhất)

*Copy toàn bộ phần dưới đây, tạo 1 Cell mới trên Kaggle và chạy:*

```python
# 1. Cài đặt các thư viện bổ trợ cho việc xuất ONNX
!pip install onnx onnxscript

# 2. Xóa thư mục cũ nếu có để tránh lỗi trùng lặp khi clone lại
import shutil
import os
if os.path.exists("MDEP-Microglial-Driven-Evidential-Pruning"):
    shutil.rmtree("MDEP-Microglial-Driven-Evidential-Pruning")

# 3. Clone mã nguồn mới nhất từ GitHub của bạn
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git

# 4. Chuyển thư mục hoạt động vào dự án
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

---

## 🏃 CELL 2: Chạy Thí Nghiệm Đơn (Chọn 1 trong các ô dưới đây và chạy)

Sau khi Cell 1 chạy xong, hãy chọn một trong các Cell dưới đây tùy thuộc vào thí nghiệm bạn muốn thực hiện:

### 🔹 Thí nghiệm 2.1: Chạy mô hình GUDS-EDL đề xuất (Full)
```python
# Đảm bảo đứng đúng thư mục chạy
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning

# Chạy mô hình đề xuất
!python experiments/isic_paper_experiments.py --experiment full_guds --epochs 15
```

### 🔹 Thí nghiệm 2.2: Chạy GroupKFold theo nhóm bệnh nhân
```python
# Đảm bảo đứng đúng thư mục chạy
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning

# Chạy GroupKFold 5 Folds
!python experiments/run_group_kfold.py --folds 5 --epochs 10 --batch_size 32
```

### 🔹 Thí nghiệm 2.3: Chạy so sánh nhiều kiến trúc Backbones
```python
# Đảm bảo đứng đúng thư mục chạy
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning

# Chạy ResNet-18, ConvNeXt-T, Swin-T
!python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 10 --batch_size 16
```

### 🔹 Thí nghiệm 2.4: Chạy so sánh các chế độ Regrowth (Calibration Study)
```python
# Đảm bảo đứng đúng thư mục chạy
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning

# Chạy so sánh các regrowth (KL, Vacuity, Ambiguity, Ratio)
!python experiments/run_calibration_study.py --modes kl_uniform vacuity ambiguity ratio --epochs 10
```

### 🔹 Thí nghiệm 2.5: Chạy xuất file ONNX thưa
```python
# Đảm bảo đứng đúng thư mục chạy
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning

# Chạy xuất file ONNX
!python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
```

### 🔹 Thí nghiệm 2.6: Chạy đánh giá Domain Shift (Smartphone vs Dermoscopic)
```python
# Đảm bảo đứng đúng thư mục chạy
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning

# Chạy kiểm định ngoại quan (External Validation)
!python experiments/run_external_validation.py
```
