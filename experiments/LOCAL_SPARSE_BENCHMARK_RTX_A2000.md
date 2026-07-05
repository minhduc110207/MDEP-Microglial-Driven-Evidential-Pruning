# Hướng dẫn chi tiết chạy Sparse Tensor Core Benchmark trên GPU RTX A2000 (Local Windows)

NVIDIA RTX A2000 (8GB/6GB) là GPU thuộc kiến trúc **Ampere** (Compute Capability 8.6, nhân đồ họa GA106). RTX A2000 được trang bị sẵn **lõi Tensor thế hệ thứ 3 (Tensor Cores)** hỗ trợ xử lý ma trận thưa 2:4 (Sparse Tensor Cores) trực tiếp ở mức phần cứng. 

Tài liệu này hướng dẫn bạn cấu hình và chạy thực nghiệm đo tốc độ trên máy Windows của mình.

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

## Phần 2: Chạy thực nghiệm PyTorch Semi-Structured Sparsity

Khi chạy script, mô hình thưa DST-EDL sẽ tự động được chuyển đổi sang định dạng nén thưa thớt của PyTorch (`to_sparse_semi_structured`) để kích hoạt thư viện tăng tốc `cuSPARSELt` của GPU.

Chạy lệnh benchmark cục bộ:
```bash
python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
```

### Kết quả hiển thị dự kiến:
1. **ONNX Export:** Tạo ra file `dst_edl_resnet18_sparse.onnx` ngay tại thư mục gốc.
2. **Dense vs Sparse:** In ra thời gian xử lý và throughput (ảnh/giây) của mô hình Dense gốc so với mô hình DST-EDL thưa.
3. **Speedup:** Hiển thị tỉ số tăng tốc thực tế (ví dụ: `Speedup: 1.25x`).

---

## Phần 3: Chạy tăng tốc toàn diện qua NVIDIA TensorRT (Local Windows)

Để tăng tốc cả các lớp Convolution (Conv2d) của mạng ResNet-18 qua phần cứng lõi Tensor, bạn cần sử dụng công cụ biên dịch `trtexec` của NVIDIA TensorRT.

### Bước 1: Tải và cài đặt TensorRT cho Windows
1. Truy cập [NVIDIA TensorRT Download](https://developer.nvidia.com/tensorrt) (Yêu cầu tài khoản NVIDIA Developer miễn phí).
2. Tải bản **TensorRT 8.6.x (hoặc 9.x) cho Windows** (tải tệp ZIP) phù hợp với phiên bản CUDA Driver máy bạn (khuyên dùng bản cho CUDA 12.x).
3. Giải nén tệp ZIP vừa tải (ví dụ giải nén vào thư mục `C:\TensorRT`).

### Bước 2: Thêm thư viện vào biến môi trường PATH của Windows
Để hệ điều hành tìm thấy file thực thi `trtexec.exe`, bạn cần thêm đường dẫn của nó vào PATH:
1. Nhấn phím `Windows` -> Tìm kiếm "env" -> Chọn **Edit the system environment variables**.
2. Nhấp vào nút **Environment Variables**.
3. Tại bảng **System variables** (phía dưới), tìm dòng `Path` -> Chọn **Edit**.
4. Nhấp vào **New** -> Thêm đường dẫn tới thư mục `lib` của thư mục vừa giải nén (ví dụ: `C:\TensorRT\lib`).
5. Nhấp vào **New** -> Thêm đường dẫn tới thư mục `bin` của thư mục vừa giải nén (ví dụ: `C:\TensorRT\bin`).
6. Chọn **OK** để lưu lại tất cả.

### Bước 3: Kiểm tra trtexec trong Command Prompt mới
**Mở một cửa sổ Terminal mới** (để cập nhật biến môi trường PATH vừa chỉnh sửa) và gõ:
```bash
trtexec --help
```
Nếu màn hình hiện danh sách hướng dẫn sử dụng của TensorRT là bạn đã cài đặt thành công.

### Bước 4: Chạy biên dịch và đo tốc độ thực tế (Level 3 Evidence)
Chạy lệnh sau để ép lõi Tensor Core trên RTX A2000 tối ưu hóa cấu trúc 2:4 thưa:
```bash
trtexec --onnx=dst_edl_resnet18_sparse.onnx --fp16 --sparsity=force --shapes=input:64x3x224x224
```

### Cách lấy số liệu cho bài báo:
Khi lệnh chạy xong, cuộn lên tìm các thông số đầu ra ở cuối log:
- **`Latency`:** Đo bằng mili-giây (ms). So sánh độ trễ trung bình của mô hình.
- **`Throughput`:** Hiển thị dưới dạng `qps` (Queries Per Second - số ảnh xử lý được mỗi giây). Bạn hãy lấy con số này để đưa vào bảng kết quả thực nghiệm tăng tốc trong paper.
