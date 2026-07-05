# Hướng dẫn chi tiết chạy Sparse Tensor Core Benchmark trên GPU RTX A2000 (Local Windows)

NVIDIA RTX A2000 (8GB/6GB) là GPU thuộc kiến trúc **Ampere** (Compute Capability 8.6, nhân đồ họa GA106). RTX A2000 được trang bị sẵn **lõi Tensor thế hệ thứ 3 (Tensor Cores)** hỗ trợ xử lý ma trận thưa 2:4 (Sparse Tensor Cores) trực tiếp ở mức phần cứng.

Tài liệu này hướng dẫn bạn cấu hình môi trường, xuất mô hình ONNX FP16 và thực hiện benchmark so sánh Dense vs Sparse chuẩn xác nhất trên **TensorRT 11**.

---

## Phần 1: Cấu hình môi trường Python & PyTorch CUDA (Local)

### Bước 1: Mở Terminal tại thư mục code
Mở PowerShell hoặc Command Prompt tại thư mục dự án của bạn (`d:\MDEP`).
*(Nếu bạn sử dụng Anaconda/Miniconda, hãy mở **Anaconda Prompt** và kích hoạt môi trường ảo của bạn trước bằng lệnh `conda activate <tên_môi_trường>`)*.

### Bước 2: Gỡ cài đặt bản PyTorch CPU cũ
Chạy lệnh sau để tránh xung đột giữa phiên bản CPU cũ và phiên bản CUDA sắp cài:
```bash
pip uninstall torch torchvision torchaudio -y
```

### Bước 3: Cài đặt PyTorch hỗ trợ GPU CUDA (NVIDIA)
Cài đặt phiên bản PyTorch CUDA 12.1 (phiên bản khuyên dùng cho dòng card RTX A2000 trên Windows):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bước 4: Kiểm tra nhận diện GPU RTX A2000
Chạy đoạn code nhỏ sau để xác thực PyTorch đã kết nối thành công với card RTX A2000:
```bash
python -c "import torch; print('CUDA hỗ trợ:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
**Yêu cầu đầu ra:**
- `CUDA hỗ trợ: True`
- `GPU Name: NVIDIA RTX A2000 8GB` (hoặc 6GB)

---

## Phần 2: Chạy thực nghiệm PyTorch Semi-Structured Sparsity (Nội bộ)

Khi chạy script, mô hình thưa DST-EDL sẽ tự động được chuyển đổi sang định dạng nén thưa thớt của PyTorch (`to_sparse_semi_structured`) để kích hoạt thư viện tăng tốc `cuSPARSELt` của GPU.

Chạy lệnh benchmark cục bộ:
```bash
python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
```

---

## Phần 3: Chạy tăng tốc toàn diện qua NVIDIA TensorRT 11 (Cực kỳ quan trọng)

Từ phiên bản **TensorRT 10 và 11**, NVIDIA áp dụng cơ chế **Strongly Typed** (Định kiểu dữ liệu nghiêm ngặt). TensorRT sẽ biên dịch mô hình chính xác theo kiểu dữ liệu được định nghĩa trực tiếp trong đồ thị ONNX. Do đó, nếu bạn xuất ONNX ở dạng FP32 thông thường, TensorRT 11 sẽ bắt buộc chạy ở chế độ FP32 và không thể kích hoạt lõi thưa vật lý (vốn chỉ hỗ trợ FP16/INT8).

Để giải quyết vấn đề này, ta cần xuất mô hình ONNX trực tiếp dưới định dạng FP16 vật lý trước khi nạp vào TensorRT.

### Bước 1: Cài đặt TensorRT 11 cho Windows
1. Tải bản **TensorRT 11.x cho Windows** (tệp ZIP) từ trang chủ NVIDIA Developer.
2. Giải nén vào một thư mục (ví dụ `D:\Dependencies\TensorRT-11.1.0.106`).
3. Thêm đường dẫn thư mục `bin` và `lib` của TensorRT vào biến môi trường `PATH` của Windows.

### Bước 2: Xuất mô hình ONNX chuẩn FP16
Chạy script xuất mô hình với tham số `--fp16`:
```bash
python experiments/export_sparse_acceleration.py --fp16 --onnx_path dst_edl_resnet18_sparse_fp16.onnx
```
Lệnh này sẽ chuyển đổi mô hình và dữ liệu giả lập về dạng Half-Precision trước khi xuất, đảm bảo đồ thị ONNX chứa các nút Float16 thuần túy.

### Bước 3: Đo đạc tốc độ tăng tốc thực tế (Level 3 Evidence)
Mở PowerShell mới và thực hiện benchmark hai chế độ chạy sau để đo đạc chênh lệch hiệu năng:

#### 1. Đo chế độ Dày (Dense FP16 - Baseline)
```bash
trtexec --onnx=dst_edl_resnet18_sparse_fp16.onnx --sparsity=disable --shapes=input:64x3x224x224
```
*Ghi lại chỉ số: Throughput (QPS) và Latency (ms).*

#### 2. Đo chế độ Thưa (MDEP/DST-EDL Sparse FP16 - Đề xuất)
```bash
trtexec --onnx=dst_edl_resnet18_sparse_fp16.onnx --sparsity=enable --shapes=input:64x3x224x224
```
*Ghi lại chỉ số: Throughput (QPS) và Latency (ms).*

---

## 📊 Bảng đối chiếu thực tế (Mẫu báo cáo khoa học)

Khi chạy hai lệnh trên trên GPU RTX A2000, bạn sẽ thu được kết quả so sánh có cấu trúc tương tự bảng dưới đây để đưa vào paper:

| Chế độ cấu hình (FP16) | Throughput (QPS) | Latency (Mean) | Tỉ số tăng tốc (Speedup) |
| :--- | :--- | :--- | :--- |
| **Dense FP16 (Baseline)** | *Ví dụ: 120.4 QPS* | *Ví dụ: 8.30 ms* | Baseline (1.00x) |
| **DST-EDL Sparse FP16 (Ours)** | *Ví dụ: 154.2 QPS* | *Ví dụ: 6.48 ms* | **1.28x** |

> **Mẹo để số liệu ổn định:** Trước khi chạy `trtexec`, hãy mở PowerShell bằng quyền Administrator và chạy lệnh khóa xung nhịp GPU để tránh hiện tượng sụt xung do nhiệt độ trên laptop:
> `nvidia-smi --lock-gpu-clocks=1200,1200`
