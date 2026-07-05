# Chạy Sparse Tensor Core 2:4 trên Kaggle

Bạn cần lưu ý một điểm vô cùng quan trọng về phần cứng trước khi chạy: **Kaggle hiện tại chỉ cung cấp GPU T4 (Turing) và P100 (Pascal)**. 
Công nghệ **Sparse Tensor Cores (tăng tốc 2:4)** của NVIDIA chỉ được phần cứng hỗ trợ bắt đầu từ kiến trúc **Ampere (Compute Capability 8.0+)** - ví dụ như GPU A100, L4, RTX 3090, RTX 4090.

Do đó, nếu bạn chạy trực tiếp trên Kaggle T4, phần cứng sẽ **không thể** cho ra tốc độ gấp đôi (2x speedup), mà nó chỉ cho phép xuất mô hình và kiểm tra tính hợp lệ (Level 1 & 2 Evidence). Để có kết quả tăng tốc thực tế (Level 3 Evidence), bạn cần mượn một GPU Ampere (ví dụ: Google Colab Pro có A100/L4, hoặc máy tính cá nhân RTX 30/40).

Dưới đây là cách bạn chạy file export và TensorRT trên Kaggle (hoặc bất kỳ môi trường nào).

## Bước 1: Chạy Script Export ONNX & PyTorch Sparse

Mở một Cell trong Kaggle Notebook (bật GPU T4) và chạy:

```python
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning

# Chạy script export. Lưu ý: Môi trường T4 sẽ cảnh báo không có GPU Ampere 
# nhưng nó vẫn sẽ xuất file ONNX thành công.
!python experiments/export_sparse_acceleration.py --batch_size 128 --image_size 224
```

Sau bước này, bạn sẽ nhận được file `dst_edl_resnet18_sparse.onnx`.

## Bước 2: Chạy TensorRT (trtexec)

Kaggle có cài sẵn TensorRT. Dù GPU T4 không hỗ trợ Sparse Tensor Core (do đó sẽ không tăng tốc 2x), nhưng bạn vẫn có thể chạy lệnh `trtexec` với cờ `--sparsity=force` để kiểm chứng rằng môi trường TensorRT hoàn toàn biên dịch thành công mô hình của bạn:

```bash
!/usr/src/tensorrt/bin/trtexec \
    --onnx=dst_edl_resnet18_sparse.onnx \
    --fp16 \
    --sparsity=force \
    --shapes=input:128x3x224x224
```
*(Đường dẫn `trtexec` có thể là `/usr/src/tensorrt/bin/trtexec` hoặc chỉ cần gọi `trtexec` tuỳ thuộc vào version container của Kaggle).*

## Khuyến nghị để có số liệu thật cho bài báo (Paper)

1. Nếu bạn bắt buộc phải cung cấp số liệu tốc độ (Images/sec) cho Reviewer, hãy thuê/dùng tạm **Google Colab (chọn GPU L4 hoặc A100)**. 
2. Chạy đúng 2 lệnh trên trong Colab (L4/A100).
3. Copy kết quả output của `trtexec` phần **Throughput** (ví dụ: `Throughput: 8500 qps`) và điền vào bảng kết quả của bài báo.
4. TensorRT trên A100/L4 sẽ tự động kích hoạt Sparse Tensor Cores khi thấy cờ `--sparsity=force` kết hợp với trọng số dạng 2:4 hợp lệ từ DST-EDL.
