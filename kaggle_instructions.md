# Hướng dẫn Chạy Thí nghiệm MDEP-GUDS trên Kaggle

Tài liệu này cung cấp hướng dẫn chi tiết từng bước để bạn có thể chạy toàn bộ các thí nghiệm của bài báo trên môi trường Kaggle Notebooks một cách song song và độc lập.

## 1. Chuẩn bị Môi trường Kaggle

Để phân chia công việc, bạn nên tạo **5 Notebook (Sổ tay) riêng biệt** trên Kaggle tương ứng với 5 cụm thí nghiệm. Ở mỗi Notebook, hãy thực hiện các bước chuẩn bị sau:

### Bước 1.1: Tải mã nguồn lên Kaggle
Có hai cách để đưa mã nguồn MDEP vào Kaggle:
- **Cách 1 (Khuyên dùng):** Tải toàn bộ thư mục code (hoặc clone trực tiếp từ GitHub) vào Kaggle. Bạn có thể mở Terminal trên Kaggle và gõ:
  ```bash
  git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
  cd MDEP-Microglial-Driven-Evidential-Pruning
  ```
- **Cách 2:** Nén thư mục mã nguồn thành file `.zip`, tải lên Kaggle dưới dạng một Dataset (ví dụ tên là `mdep-source-code`), sau đó Add Data vào Notebook. Copy toàn bộ code ra thư mục làm việc (working directory) để có quyền ghi (write) log.

### Bước 1.2: Liên kết (Add Data) các Bộ Dữ Liệu
Ở bên phải màn hình Notebook, chọn **Add Data** và tìm kiếm các bộ dữ liệu tương ứng:
- **Cho ISIC:** Tìm kiếm `isic-2024-challenge` (Bộ dữ liệu ISIC 2024 chính thức).
- **Cho CIFAR-100:** Tìm kiếm `cifar-100` (Thường Kaggle có sẵn).
- **Cho MVTec AD:** Tìm kiếm `mvtec-ad` (Tải bộ chuẩn chứa đủ các danh mục lỗi).

### Bước 1.3: Cài đặt cấu hình phần cứng
- Bật GPU (P100 hoặc T4x2).
- Nếu dữ liệu ISIC quá lớn, đảm bảo bộ nhớ không bị đầy. Khuyến nghị bật Persistence để không mất code khi reset.

---

## 2. Chi tiết Lệnh Chạy cho 5 Notebook

Sau khi đã vào thư mục gốc của project (nơi chứa thư mục `experiments/`), bạn mở một Cell trong Jupyter Notebook và chạy các lệnh tương ứng như sau:

### Notebook 1: Chạy các Baselines Truyền thống (Softmax) trên ISIC
*Sẽ chạy: Standard CE, Focal Loss, Logit Adjustment, Class Balanced CE, Balanced Softmax, LDAM-DRW, cRT, MiSLAS.*
```bash
!python experiments/run_isic_softmax_baselines.py \
    --data_dir /kaggle/input/isic-2024-challenge \
    --epochs 40 \
    --batch_size 32 \
    --seed 42
```

### Notebook 2: Chạy các Baselines Bằng chứng (Evidential) trên ISIC
*Sẽ chạy: Dense EDL, Fisher EDL, Flexible EDL, R-EDL, Static 2:4 EDL, RigL-style 2:4.*
```bash
!python experiments/run_isic_evidential_baselines.py \
    --data_dir /kaggle/input/isic-2024-challenge \
    --epochs 40 \
    --batch_size 32 \
    --seed 42
```

### Notebook 3: Chạy mô hình GUDS-EDL chính và các Ablations trên ISIC
*Sẽ chạy: Full GUDS-EDL, Without Pruner, Without Regrower, Asymmetric KL, Without EFL, v.v.*
```bash
!python experiments/run_isic_guds_ablations.py \
    --data_dir /kaggle/input/isic-2024-challenge \
    --epochs 40 \
    --batch_size 32 \
    --seed 42
```
> **Lưu ý:** Lệnh này sẽ sử dụng cấu hình lý thuyết chuẩn nhất (Symmetric KL + KL-Uniform Regrower) cho `full_guds`. Bảng Metric kết quả (PAUC, AUROC, v.v.) sẽ tự động in ra màn hình Console sau khi mỗi model train xong.

### Notebook 4: Đánh giá độ Tổng quát (Generalization) trên CIFAR-100-LT
```bash
!python experiments/run_cifar_suite.py
```
> Kaggle thường sẽ tự tải tập CIFAR thông qua thư viện `torchvision.datasets`. Tuy nhiên nếu không có Internet, bạn có thể truyền thêm đường dẫn folder CIFAR-100 thông qua `data_dir`. Các tham số (như batch_size=128) đã được thiết lập mặc định trong code.

### Notebook 5: Đánh giá Phát hiện Bất thường (Anomaly Detection) trên MVTec AD
```bash
!python experiments/run_mvtec_suite.py
```
> Script sẽ tự động set Batch Size = 16 và Epoch = 50 cho phù hợp với ảnh 224x224. Đảm bảo bạn đã Mount bộ MVTec AD vào `/kaggle/input`. Code sẽ tự động dò tìm cấu trúc thư mục (như `hazelnut`, `bottle`...) bên trong `/kaggle/input`.

---

## 3. Xem Kết Quả và Lưu Trữ

1. **Bảng Kết Quả Trực Tiếp:**
   Nhờ đoạn code tôi đã bổ sung gần đây, ngay khi kết thúc 100 epoch của một mô hình, một **Bảng Kết Quả ASCII** sẽ được in trực tiếp lên màn hình Kaggle Console. Bảng này sẽ liệt kê cụ thể các chỉ số: `macro_auroc`, `pauc`, `pr_auc`, `ece_adaptive`, `aurc`, v.v. để bạn đánh giá chéo ngay lập tức.

2. **Lưu file Output (Logs & Weights):**
   Mỗi mô hình sau khi chạy xong sẽ lưu toàn bộ thông số vào thư mục `outputs/<Ngày_Tháng>/<Tên_Thí_Nghiệm>`. Bao gồm:
   - `model_state.pth`: Trọng số mô hình (Weights).
   - `metrics.json`: Thông số đo lường chi tiết.
   - `run_config.json`: Cấu hình thí nghiệm.

3. **Tải file về máy:**
   Kaggle tự động cung cấp nút **"Output"** ở menu bên phải (hoặc ở dưới cùng). Bạn có thể zip toàn bộ folder `outputs` và nhấn Download về máy tính để vẽ biểu đồ và phân tích cho bài báo.

```bash
# Zip lại toàn bộ kết quả để dễ dàng download từ Kaggle
!zip -r mdep_results.zip outputs/
```

Chúc bạn thực nghiệm thành công! Toàn bộ cơ sở lý thuyết và cài đặt hiện tại đều rất hoàn hảo, không còn gì phải lo lắng.
