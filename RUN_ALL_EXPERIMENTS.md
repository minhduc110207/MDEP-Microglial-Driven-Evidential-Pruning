# Hướng Dẫn Chạy Toàn Bộ Thí Nghiệm (Cheatsheet)

Tài liệu này tổng hợp tất cả các câu lệnh chạy các thí nghiệm trong đề tài nghiên cứu **MDEP (Microglial-Driven Evidential Pruning)**. 

Bạn có thể chạy các lệnh này trực tiếp tại thư mục gốc của dự án (`d:\MDEP` trên máy cá nhân, hoặc `/kaggle/working/MDEP-...` trên Kaggle Notebook).

---

## 📌 1. Chạy Thí Nghiệm ISIC 2024 Chính (Bài báo chính)
Sử dụng script chính `experiments/isic_paper_experiments.py`.

*   **Chạy thuật toán đề xuất GUDS-EDL (Full):**
    ```bash
    python experiments/isic_paper_experiments.py --experiment full_guds
    ```
*   **Chạy Baseline thưa RigL-style 2:4:**
    ```bash
    python experiments/isic_paper_experiments.py --experiment rigl_style_24
    ```
*   **Chạy các Baseline Dense (CE, Focal Loss, Balanced Softmax, v.v.):**
    ```bash
    python experiments/isic_paper_experiments.py --experiment standard_ce
    python experiments/isic_paper_experiments.py --experiment focal_loss
    python experiments/isic_paper_experiments.py --experiment balanced_softmax
    ```
*   **Chạy toàn bộ các nhóm thí nghiệm cùng lúc:**
    ```bash
    python experiments/isic_paper_experiments.py --suite main_tables
    ```
*   **Tham số hữu ích:**
    - `--allow_dummy_data`: Sử dụng khi chạy thử (dry-run) trên máy không có dữ liệu ISIC gốc.
    - `--epochs <N>`: Đặt số lượng epochs (ví dụ: `--epochs 15`).
    - `--save_model`: Lưu file trạng thái mô hình `.pth`.

---

## 📌 2. Chạy GroupKFold theo Nhóm Bệnh Nhân (Patient Clustering)
Kiểm tra độ vững chãi của thuật toán trên các phân tách dữ liệu khác nhau nhằm ngăn chặn rò rỉ dữ liệu.

*   **Chạy GroupKFold 5-Fold trên dữ liệu thật:**
    ```bash
    python experiments/run_group_kfold.py --folds 5 --epochs 15 --batch_size 32
    ```
*   **Chạy thử nghiệm nhanh (Dry-run):**
    ```bash
    python experiments/run_group_kfold.py --folds 5 --allow_dummy_data --epochs 2
    ```

---

## 📌 3. Đánh Giá Đa Dạng Backbones (Độc lập Kiến Trúc)
Kiểm thử DST-EDL trên các mạng khác ngoài ResNet-18 bao gồm Swin Transformer và ConvNeXt.

*   **Chạy benchmark mở rộng Backbones:**
    ```bash
    python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 15
    ```
*   **Chạy thử nghiệm nhanh (Dry-run):**
    ```bash
    python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --allow_dummy_data --epochs 2
    ```

---

## 📌 4. So Sánh Cơ Chế Regrowth (DST-EDL vs RigL)
Đo lường và so sánh hiệu quả của việc tìm kiếm Topology thưa dựa trên độ bất định Evidential Dirichlet.

*   **Chạy lệnh so sánh trực quan và xuất bảng kết quả:**
    ```bash
    python experiments/compare_rigl_guds.py
    ```

---

## 📌 5. Nghiên Cứu Calibration & Cắt Bỏ Regrowth (Ablation Study)
Nghiên cứu độ chính xác lâm sàng thông qua các chỉ số Brier Score, ECE, e-AURC và so sánh các biến thể regrowth.

*   **Chạy đầy đủ các chế độ Regrowth (KL, Vacuity, Ambiguity, Ratio):**
    ```bash
    python experiments/run_calibration_study.py --modes kl_uniform vacuity ambiguity ratio --epochs 15
    ```
*   **Chạy thử nghiệm nhanh (Dry-run):**
    ```bash
    python experiments/run_calibration_study.py --allow_dummy_data --epochs 2
    ```

---

## 📌 6. Tăng Tốc Phần Cứng & ONNX Export (RTX A2000 / Colab L4)
Xuất mô hình thưa sang định dạng ONNX và biên dịch tối ưu hóa qua NVIDIA TensorRT.

*   **Xuất mô hình sang ONNX và đo tốc độ thưa trên GPU nội bộ:**
    ```bash
    python experiments/export_sparse_acceleration.py --batch_size 64 --image_size 224
    ```
*   **Lệnh biên dịch TensorRT (Chạy trên Terminal của máy có cài TensorRT SDK):**
    ```bash
    trtexec --onnx=dst_edl_resnet18_sparse.onnx --fp16 --sparsity=force --shapes=input:64x3x224x224
    ```

---

## 📌 7. Đánh Giá Khả Năng Tổng Quát Hóa (Fitzpatrick17k & PAD-UFES-20)
Đo lường độ sụt giảm hiệu năng khi gặp Domain Shift, đo khoảng cách công bằng (EOM gap) và khả năng phát hiện ảnh bất thường (OOD).

*   **Chạy đánh giá tổng quát hóa:**
    ```bash
    python experiments/run_external_validation.py
    ```
*   **Đánh giá checkpoint mô hình cụ thể của bạn:**
    ```bash
    python experiments/run_external_validation.py --model_path paper_experiment_outputs/isic/full_guds/seed_42/model_state.pth
    ```

---

## 📂 Nơi lưu kết quả đầu ra (Outputs)
Sau khi chạy các lệnh trên, kết quả dạng JSON chứa toàn bộ chỉ số sẽ tự động được ghi nhận tại:
- `./paper_experiment_outputs/isic/` (Dành cho ISIC)
- `./paper_experiment_outputs/backbones/` (Dành cho Backbone)
- `./paper_experiment_outputs/group_kfold_summary.json` (Dành cho GroupKFold)
- `./paper_experiment_outputs/calibration_ablation_summary.json` (Dành cho Ablations)
- `./paper_experiment_outputs/external_validation_summary.json` (Dành cho Domain Shift)
