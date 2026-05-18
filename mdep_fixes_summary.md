# Tổng Hợp Các Sửa Lỗi và Tối Ưu Hóa Hệ Thống MDEP

Tài liệu này tổng hợp toàn bộ các vấn đề (bug) về logic, toán học và kỹ thuật tối ưu hóa trong quá trình huấn luyện mô hình Evidential Deep Learning (EDL) mà chúng ta đã rà soát và sửa chữa trên hệ thống MDEP. 

Các thay đổi đã được đồng bộ trên toàn bộ các tệp mã nguồn: `main.py`, `losses.py`, `trainer.py`, `mdep_agents.py`, `mdep_notebook.py`, `mdep_ablation_prune_only.py`, và `mdep_ablation_grow_only.py`.

---

## 1. Ngắt Gradient của Hệ số Tiêu cự (Focal Weight Detachment)
**Vấn đề:** 
Hệ số tiêu cự $(1-\hat{p}_c)^\gamma$ chứa xác suất dự đoán $\hat{p}_c = \alpha_c / S$. Nếu để mặc định, PyTorch sẽ lan truyền ngược qua hệ số này, phá vỡ định dạng phân phối Dirichlet và khiến bằng chứng bị đẩy về 0 một cách dị thường.
**Giải pháp:** 
Thêm `.detach()` vào biểu thức tính toán `focal_weight` trong `EvidentialFocalLoss`.
- **Mã nguồn:** `focal_weight = (1.0 - p_target.detach()) ** self.gamma`

## 2. Ủ nhiệt cho KL Divergence (KL Annealing)
**Vấn đề:** 
Nếu áp dụng toàn bộ sức mạnh của hàm phạt KL ngay từ epoch đầu tiên (khi mô hình chưa học được gì), mạng sẽ bị ép toàn bộ bằng chứng về 0 để tránh bị phạt.
**Giải pháp:** 
Bổ sung tham số `epoch` vào hàm `forward` của `EvidentialFocalLoss` và tính toán một hệ số nhân `annealing_coef` tăng dần từ 0 lên 1 trong 10-15 epoch đầu tiên.
- **Mã nguồn:** `annealing_coef = min(1.0, epoch / self.annealing_epochs)`

## 3. Sửa Lỗi Toán Học của Tác Nhân Astrocyte (Trong bản Multi-file)
**Vấn đề:** 
Ở bản chia nhỏ file (`trainer.py`, `mdep_agents.py`), gradient của tác nhân Astrocyte $u_{e, i}^{(node)}$ đang bị tính toán sai lầm bằng cách lấy `u_e.backward()` trực tiếp trên ma trận trọng số. Nó không đo lường được "neuron mù" (blind neurons) dựa trên activations (kích hoạt) như trong lý thuyết bài báo.
**Giải pháp:** 
Thay thế logic tính toán bằng **Forward Hooks** để lưu trữ tensor kích hoạt $a_i^{(l)}$. Sau đó dùng `torch.autograd.grad(u_e, act_tensors)` để lấy đạo hàm của độ bất định nhận thức theo từng node, sau đó broadcast thành dạng ma trận (đã đồng bộ với bản `mdep_notebook.py`).

## 4. Tắc Nghẽn Hàm Kích Hoạt (Activation Bottleneck) - Đã xác minh an toàn
**Vấn đề:** 
ReLU có thể gây chết neuron (bằng chứng = 0 triệt để) do cắt gradient.
**Tình trạng:** 
Mã nguồn của bạn đã được thiết lập rất tốt từ trước: lớp `EvidenceLayer` luôn nhận `activation='softplus'`, đảm bảo gradient luôn mượt mà. Đã xác minh hoàn toàn an toàn.

## 5. Bất Ổn Định Số Học (log(0) / NaN) - Đã xác minh an toàn
**Vấn đề:** 
Các hàm logarit và digamma có thể bị sập (crash) nếu giá trị đầu vào bằng 0.
**Tình trạng:** 
Toán học nguyên bản của EDL đã cộng thêm hằng số 1 (`alpha = evidence + 1.0`). Do `evidence >= 0`, `alpha` luôn $\ge 1.0$. Các hàm `digamma(1.0)` và `lgamma(1.0)` hoàn toàn hợp lệ và không sinh ra -Inf. Do đó hệ thống này miễn nhiễm với lỗi `log(0)`.

## 6. Xung Đột Với Label Smoothing - Đã xác minh an toàn
**Vấn đề:** 
EDL yêu cầu nhãn one-hot thuần túy (tuyệt đối không dùng label smoothing như [0.9, 0.1]).
**Tình trạng:** 
Mã nguồn hàm mất mát đã chủ động dùng `targets = F.one_hot(targets, num_classes=self.num_classes).float()`. Do đó nhãn được ép buộc là one-hot nguyên chất, chặn đứng mọi kỹ thuật label smoothing từ Dataloader.

## 7. Khởi Tạo Trọng Số Đầu Ra (Weight Initialization)
**Vấn đề:** 
Khởi tạo trọng số ngẫu nhiên Kaiming/Xavier cho lớp Softplus cuối có thể tạo ra bằng chứng (evidence) siêu lớn ban đầu (ví dụ $e=100$). KL Divergence sẽ đánh dấu đây là một sự "tự tin thái quá" cực kì nghiêm trọng và đập nát toàn bộ trọng số mạng lưới bằng một cú gradient bùng nổ.
**Giải pháp:** 
Bổ sung đoạn mã ép trọng số của lớp Linear cuối cùng về phân phối chuẩn với mean=0 và std cực nhỏ (0.001) trong tất cả các script.
- **Mã nguồn:** `nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)`

## 8. Bù Đắp Mất Mát Focal (Focal Loss Scaling) và Lịch Trình Warmup Học Suất
**Vấn đề:** 
Hàm loss luôn bị "đóng băng" ở mức 0.2. Hệ số Focal $(1-\hat{p}_t)^\gamma$ khiến hàm loss bị bóp nghẹt 4 lần ở giai đoạn đầu. Quanh điểm $e=0$, phân phối Dirichlet là một vùng cực kì bằng phẳng. Các gradient truyền xuống quá li ti khiến Optimizer AdamW không thể lấy được mô-men (momentum) để vọt qua.
**Giải pháp:** 
- **Loss Scaling:** Chủ động nhân hàm loss đầu ra với một hằng số tỷ lệ (ở đây là `4.0`) để trung hòa sức ép của hệ số tiêu cự.
- **Manual LR Warmup:** Khởi tạo học suất từ con số cực nhỏ ($1e^{-6}$) và tăng tịnh tiến lên Base LR ($1e^{-3}$) trong 5 epoch đầu tiên. Điều này giúp Adam tích lũy được đà một cách tự nhiên.
- **Gradient Clipping & Logging:** Kẹp (clip) norm của gradient tại 2.0 để chống nổ, đồng thời in cả `LR` và `GradNorm` ra màn hình mỗi vòng lặp để bạn trực tiếp giám sát quá trình "vượt ngục" của mô hình khỏi vùng zero-evidence.
