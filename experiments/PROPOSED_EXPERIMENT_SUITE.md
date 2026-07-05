# Thiết Kế Chi Tiết Hệ Thống Thực Nghiệm Bổ Sung cho DST-EDL

Tài liệu này thiết kế chi tiết 7 bài thử nghiệm (experiments) mới cùng hệ thống chỉ số (metrics) thích hợp nhằm củng cố tính thuyết phục khoa học của thuật toán DST-EDL khi nộp báo cáo (paper) cho AAAI.

---

## Bảng Tổng Hợp Thiết Kế & Chỉ Số (Metrics) Khuyên Dùng

| # | Thí nghiệm | Thiết kế kỹ thuật (Methodology) | Chỉ số đo lường (Metrics) | Ý nghĩa đóng góp cho Paper |
|---|---|---|---|---|
| **1** | **Multi-seed / Multi-split theo Patient Groups** | Chia tập dữ liệu ISIC 2024 theo nhóm bệnh nhân (`patient_id`). Chạy 5-Fold Cross-Validation đảm bảo ảnh của cùng 1 bệnh nhân không nằm ở cả tập train và val. | - Mean & Std của pAUC$_{0.80}$<br>- Macro AUROC<br>- Balanced Accuracy (BACC) | Chứng minh mô hình bền bỉ (robust), không bị overfitting hoặc nhiễu do phân bố bệnh nhân (nhất là khi tỷ lệ dương tính cực hiếm). |
| **2** | **Thử nghiệm trên các Backbones khác** | Áp dụng wrapper DST-EDL lên các kiến trúc: Swin Transformer (ViT), ConvNeXt, và EdgeNext (backbone của đội vô địch ISIC). | - pAUC$_{0.80}$ và Macro AUROC<br>- Số lượng Params & MACs thực tế<br>- Valid 2:4 block fraction | Chứng minh DST-EDL là giải pháp tổng quát (general framework) độc lập với kiến trúc mạng (backbone-agnostic). |
| **3** | **So sánh với Evidential RigL-style 2:4** | Thay thế bộ điều khiển regrowth dựa trên độ bất định (uncertainty-driven) của DST-EDL bằng thuật toán hồi sinh dựa trên độ lớn gradient (magnitude/gradient-driven) như RigL truyền thống, nhưng vẫn giữ nguyên Evidential Head. | - Tốc độ hội tụ (AUROC vs Epoch)<br>- Final pAUC$_{0.80}$<br>- Tỷ lệ trùng lặp Topology | Tách biệt đóng góp của lõi Evidential Uncertainty khỏi cơ chế cắt tỉa thưa (DST) thuần túy. Trả lời câu hỏi: *"EDL signal thực sự có ích cho Topology?"* |
| **4** | **Nghiên cứu sâu về Calibration (Độ hiệu chuẩn)** | Đánh giá độ khớp giữa xác suất dự đoán của mô hình và thực tế lâm sàng thông qua các kỹ thuật Post-hoc Calibration (như Temperature Scaling). | - ECE (Expected Calibration Error)<br>- Brier Score / Negative Log-Likelihood (NLL)<br>- Area Under Risk-Coverage (AURC)<br>- e-AURC (Excess AURC) | Chỉ ra sự cân bằng (trade-off) giữa độ nhạy (Sensitivity) và độ hiệu chuẩn (ECE), chứng minh tính khả thi của mô hình trong việc cho phép từ chối chẩn đoán (reject option). |
| **5** | **Ablation trên từng thành phần Uncertainty** | Huấn luyện các biến thể của DST-EDL bằng cách chỉ sử dụng: (a) Vacuity (độ bất định do thiếu dữ liệu), (b) Dissonance / Ambiguity (độ bất định do xung đột thông tin), hoặc (c) Tỷ lệ kết hợp. | - Tốc độ regrowth và độ đa dạng Topology<br>- pAUC$_{0.80}$<br>- ECE | Định lượng chính xác nguồn gốc mang lại hiệu năng cho DST-EDL. Xác nhận thành phần bất định nào đóng vai trò chính. |
| **6** | **Thử nghiệm Export và chạy Hardware thật** | Sử dụng script `export_sparse_acceleration.py` xuất mô hình sang ONNX, nạp vào TensorRT chạy trên GPU RTX A2000 / A100. | - Latency (Độ trễ trung bình - ms)<br>- Throughput (Images/sec)<br>- Tốc độ tăng tốc thực tế (Speedup x) | Chuyển đổi từ khẳng định "khả thi thưa thớt" (Level 1/2) sang kết quả "tăng tốc phần cứng thực tế" (Level 3). |
| **7** | **External Validation (Domain Shift)** | Huấn luyện mô hình trên ISIC 2024, đánh giá trực tiếp (Zero-shot) trên các tập dữ liệu da liễu lâm sàng bên ngoài như **Fitzpatrick17k** và **PAD-UFES-20**. | - Cross-dataset AUROC / Macro-F1<br>- EOM (Equalized Odds Metric) theo nhóm da<br>- OOD Detection AUROC | Đo lường khả năng chống chịu sự thay đổi thiết bị chụp (lâm sàng smartphone vs dermoscopic) và tính công bằng đối với các màu da khác nhau. |

---

## Chi Tiết Hướng Dẫn Kỹ Thuật Cho Từng Bài Thí Nghiệm

### Thí nghiệm 1: Patient-Group Multi-Split
- **Mã nguồn liên quan:** `experiments/isic_paper_experiments.py`
- **Cách thực hiện:** Thay vì chia train/val ngẫu nhiên theo chỉ mục hàng (row index), hãy sử dụng `sklearn.model_selection.GroupKFold` với nhóm là cột `patient_id`. 
- **Chỉ số:** Tính toán phương sai (Standard Deviation) trên 5 fold. Một độ lệch chuẩn nhỏ về pAUC ($< 0.01$) chứng minh DST-EDL không bị ảnh hưởng bởi nhiễu cụm bệnh nhân (patient clustering).

### Thí nghiệm 2: Đa dạng hóa Backbone
- **Mã nguồn liên quan:** `Vision Transformer backbone/src/swin_main.py`
- **Cách thực hiện:** Đối với Swin Transformer hoặc ConvNeXt, áp dụng wrapper `replace_conv2d_with_mdep` và `replace_linear_with_mdep`. Đảm bảo các chiều input/output của các lớp Linear trong khối Attention/MLP chia hết cho 8 và 16 để đạt chuẩn 2:4 thưa của Tensor Cores.

### Thí nghiệm 3: So sánh với Evidential RigL (RigL vs DST-EDL)
- **Cách thực hiện:** Xây dựng một baseline huấn luyện thưa động mà trong đó:
  - Pruning: Dựa trên độ lớn trọng số $|W|$ (như thông thường).
  - Regrowth: Thay vì lấy mẫu theo phân phối độ bất định Dirichlet, ta lấy mẫu theo độ lớn gradient tích lũy của các trọng số đã bị tỉa (Magnitude Gradient Regrow).
- **Chỉ số:** Đo tốc độ cải thiện của pAUC$_{0.80}$ qua từng epoch. Nếu DST-EDL đạt độ chính xác cao hơn ở các epoch đầu, điều đó chứng minh bộ điều khiển độ bất định Dirichlet giúp định hướng Topology nhanh hơn việc dò tìm bằng gradient.

### Thí nghiệm 4: Phân Tích Độ Hiệu Chuẩn (Calibration & Selective Classification)
- **Mã nguồn liên quan:** `experiments/metrics_ext.py` -> Hàm `binary_image_anomaly_metrics` và `failure_detection_metrics`.
- **Cách thực hiện:** 
  - Sử dụng chỉ số **e-AURC (Excess Area Under Risk-Coverage Curve)**: Tính bằng diện tích dưới đường cong sai số-độ phủ của mô hình trừ đi đường cong của mô hình Oracle hoàn hảo. Chỉ số e-AURC càng nhỏ chứng minh độ tự tin của mô hình càng phản ánh đúng độ chính xác thực tế.
  - Vẽ đồ thị **Risk-Coverage Curve**: Trục X là tỷ lệ mẫu giữ lại để chẩn đoán (Coverage từ 0% đến 100%), trục Y là tỷ lệ chẩn đoán sai trên tập đó (Risk).
  - Áp dụng **Post-hoc Temperature Scaling**: Tìm một hệ số nhiệt độ $T$ tối ưu trên tập Validation để làm mịn xác suất Dirichlet trước khi tính toán ECE trên tập Test.

### Thí nghiệm 5: Ablation trên Vacuity và Ambiguity
- **Cách thực hiện:** 
  - Chạy mô hình chỉ với tín hiệu dẫn đường là **Vacuity (độ thưa thớt bằng chứng)**: $u_v = K / S$.
  - Chạy mô hình chỉ với **Ambiguity (xung đột thông tin)**: Đo bằng Entropy của Dirichlet.
- **Chỉ số:** So sánh xem biến thể nào giữ được biểu diễn đặc trưng (representation) tốt nhất mà không bị sụp đổ cấu trúc (representation collapse).

### Thí nghiệm 6: Biên Dịch Sang Hardware Thật
- **Mã nguồn liên quan:** `experiments/export_sparse_acceleration.py` và tài liệu hướng dẫn chạy cục bộ `LOCAL_SPARSE_BENCHMARK_RTX_A2000.md`.
- **Cách thực hiện:** Đo throughput trên máy local của bạn (sau khi cài CUDA) hoặc trên Colab GPU Ampere. Ghi nhận tốc độ xử lý (fps) thực tế.

### Thí nghiệm 7: Đánh Giá Domain Shift ( Fitzpatrick17k & PAD-UFES-20 )
- **Cách thực hiện:** 
  - Lấy mô hình ResNet-18 đã được train thưa trên ISIC 2024.
  - Chạy suy luận trực tiếp (Zero-shot inference) trên tập dữ liệu **PAD-UFES-20** (ảnh chụp bằng điện thoại có độ nhiễu ánh sáng lớn) và **Fitzpatrick17k** (phân loại theo thang đo màu da Fitzpatrick từ 1 đến 6).
  - **Chỉ số EOM (Equalized Odds Metric):** Đo sự chênh lệch về True Positive Rate (TPR) giữa các nhóm màu da ( Fitzpatrick 1-2 vs Fitzpatrick 5-6 ). Khoảng cách EOM càng gần 0 chứng minh thuật toán thưa DST-EDL hoạt động công bằng, không bị thiên vị chủng tộc/màu da do nhiễu nền của ảnh dermoscopic.
