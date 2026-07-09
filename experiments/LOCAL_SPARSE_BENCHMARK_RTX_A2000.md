# NVIDIA TensorRT Level-3 benchmark trên RTX A2000

Tài liệu này chạy tăng tốc sparse thật bằng TensorRT. Nó không thay thế
`hardware_profile.py`: script cũ vẫn dùng để báo structural density, exact 2:4,
MAC/FLOP lý thuyết và masked-PyTorch diagnostics.

## 1. Điều kiện bắt buộc

- Windows hoặc Linux với NVIDIA RTX A2000.
- Driver NVIDIA hoạt động (`nvidia-smi`).
- PyTorch CUDA dùng được.
- TensorRT có `trtexec.exe`/`trtexec`.
- Python packages: `torch`, `torchvision`, `onnx`.
- Bốn checkpoint fair-v3 NVIDIA-layout đã huấn luyện:

```text
paper_experiment_outputs/isic/dense_edl_fair_v3_nvidia24/seed_42/model_state.pth
paper_experiment_outputs/isic/static_24_edl_fair_v3_nvidia24/seed_42/model_state.pth
paper_experiment_outputs/isic/rigl_style_24_fair_v3_nvidia24/seed_42/model_state.pth
paper_experiment_outputs/isic/full_guds_fair_v3_nvidia24/seed_42/model_state.pth
```

Không dùng `--allow_legacy_checkpoint` cho số liệu đưa vào bài báo.

## 2. Chạy trên Windows PowerShell

Nếu `trtexec.exe` đã có trong `PATH`:

```powershell
.\experiments\run_nvidia_hardware_rtx_a2000.ps1
```

Nếu TensorRT nằm ở thư mục riêng:

```powershell
.\experiments\run_nvidia_hardware_rtx_a2000.ps1 `
  -TrtExec "D:\Dependencies\TensorRT-11.1.0.106\bin\trtexec.exe" `
  -Seed 42 `
  -Repeats 5 `
  -DurationSeconds 10 `
  -BatchSizes 1,64
```

Hoặc gọi Python trực tiếp:

```powershell
python experiments\nvidia_sparse_benchmark.py `
  --trtexec "D:\Dependencies\TensorRT-11.1.0.106\bin\trtexec.exe" `
  --seed 42 `
  --batch_sizes 1 64 `
  --repeats 5 `
  --warmup_ms 2000 `
  --duration_s 10
```

Có thể ghi đè checkpoint:

```powershell
python experiments\nvidia_sparse_benchmark.py `
  --trtexec "D:\TensorRT\bin\trtexec.exe" `
  --checkpoint "dense_edl=D:\checkpoints\dense\model_state.pth" `
  --checkpoint "static_24_edl=D:\checkpoints\static\model_state.pth" `
  --checkpoint "rigl_style_24=D:\checkpoints\rigl\model_state.pth" `
  --checkpoint "full_guds=D:\checkpoints\guds\model_state.pth"
```

## 3. Runner thực sự đo gì?

Runner tạo hai phép so sánh độc lập:

1. **Network comparison:** Dense EDL chạy TensorRT dense; Static 2:4, RigL 2:4
   và DST-EDL cho phép TensorRT chọn sparse tactics.
2. **Kernel ablation:** cùng một frozen DST-EDL ONNX chạy với
   `--sparsity=disable` và `--sparsity=enable`. Đây là phép so sánh sạch nhất để
   kết luận sparse kernel có tăng tốc hay không.

Mỗi checkpoint được:

- kiểm tra `isic_fair_v3_nvidia24_2026_07_09`;
- xác nhận exact 2:4;
- đóng băng mask đúng một lần;
- loại controller MDEP và mask multiplication khỏi inference graph;
- kiểm tra output trước/sau đóng băng;
- xuất ONNX và chạy `onnx.checker`;
- build TensorRT FP16 engine;
- benchmark bằng CUDA Graph, không tính host-device transfer;
- lặp lại theo thứ tự ngẫu nhiên để giảm order/thermal bias.

## 4. Quality gates

TensorRT chỉ được coi là dùng sparse path khi verbose build log có dạng:

```text
(Sparsity) Found N layer(s) eligible to use sparse tactics
(Sparsity) Chose M layer(s) using sparse tactics
```

`M` phải lớn hơn 0. Chỉ đặt `--sparsity=enable` không đủ làm bằng chứng.

File `paper_table.tex` chỉ được sinh khi:

- GPU đúng RTX A2000;
- cả bốn checkpoint đúng fair-v3 NVIDIA-layout;
- graph trước/sau đóng băng tương đương;
- đủ số repeat;
- mọi sparse engine được báo cáo thực sự chọn ít nhất một sparse tactic.

## 5. Output

Mỗi lần chạy tạo một thư mục timestamp dưới:

```text
paper_experiment_outputs/hardware_nvidia_rtx_a2000/
```

Các file quan trọng:

- `results.json`: manifest, môi trường, quality gates và toàn bộ kết quả.
- `summary.csv`: mean/median/std của throughput và latency.
- `raw_repeats.csv`: dữ liệu từng repeat.
- `logs/build_*.log`: bằng chứng TensorRT eligible/chosen sparse layers.
- `logs/run_*.log`: log throughput/latency gốc.
- `paper_table.tex`: chỉ có khi toàn bộ quality gates đạt.

Không sửa số trong `final (1).tex` trước khi `paper_ready=true`.
