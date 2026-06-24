# Ghi chú thay đổi lý thuyết trong `main_text.tex`

Tài liệu này ghi lại các thay đổi đã thực hiện để phần lý thuyết của `main_text.tex` gọn hơn trong main body nhưng vẫn đầy đủ cơ sở toán học trong appendix. Mục tiêu là làm bản thảo phù hợp hơn với phong cách AAAI: phần chính trình bày định nghĩa, cơ chế, mệnh đề và ý nghĩa; các chứng minh dài được đưa xuống phụ lục.

## 1. Tái cấu trúc phần chứng minh

Thay đổi quan trọng nhất là toàn bộ các khối `proof` dài trong main body đã được chuyển xuống Appendix A.

Trong main body, các mệnh đề vẫn được giữ lại để reviewer nhìn thấy rằng phương pháp có bảo đảm toán học rõ ràng. Tuy nhiên, phần chứng minh chi tiết được đặt trong:

```tex
\section{Appendix A: Proofs and Mathematical Derivations}
\label{app:proofs}
```

Cách tổ chức này giúp phần chính ngắn hơn, dễ fit vào giới hạn 7 trang của AAAI, trong khi vẫn giữ đầy đủ lập luận toán học trong phụ lục.

## 2. Chuẩn hóa ký hiệu positive part

Trước đó công thức pruning dùng `ReLU`:

```tex
\operatorname{ReLU}\left(w_{ij}\frac{\partial R}{\partial w_{ij}}\right)
```

Đã đổi thành ký hiệu toán học:

```tex
\left[w_{ij}\frac{\partial R}{\partial w_{ij}}\right]_+
```

với định nghĩa:

```tex
[x]_+ = \max(x,0)
```

Lý do: ở đây ta không dùng ReLU như một activation layer của neural network, mà dùng toán tử lấy phần dương để lọc các connection có first-order removal effect có lợi. Viết bằng `[x]_+` làm công thức trông toán học hơn và tránh cảm giác phương pháp “gắn ReLU tùy ý”.

## 3. Proposition: Validity of Evidential State

Mệnh đề này đảm bảo rằng cách tham số hóa evidence tạo ra một trạng thái Dirichlet hợp lệ.

Ta có logits:

```tex
\bm{z}\in\mathbb{R}^K
```

Evidence được tính bằng Softplus:

```tex
e_c = \ln(1+\exp(z_c))
```

Vì Softplus luôn dương với logits hữu hạn, nên:

```tex
e_c > 0
```

Dirichlet concentration parameter là:

```tex
\alpha_c = e_c + 1
```

Do đó:

```tex
\alpha_c > 1
```

Tổng evidence/Dirichlet strength:

```tex
S = \sum_{c=1}^{K}\alpha_c
```

vì mọi `\alpha_c > 1`, nên:

```tex
S > K
```

Expected probability:

```tex
\hat{p}_c = \frac{\alpha_c}{S}
```

thỏa:

```tex
\hat{p}_c > 0,\qquad \sum_{c=1}^{K}\hat{p}_c = 1
```

vì vậy:

```tex
\hat{\bm{p}}\in\Delta^K
```

tức là vector xác suất nằm trên simplex hợp lệ.

Vacuity:

```tex
u_e = \frac{K}{S}
```

Vì `S>K`, nên với logits hữu hạn:

```tex
u_e\in(0,1)
```

Giá trị 1 chỉ là supremum, đạt tới trong giới hạn khi mọi evidence tiến về 0.

Expected categorical entropy:

```tex
u_a =
\sum_{c=1}^{K}
\frac{\alpha_c}{S}
\left[
\psi(S+1)-\psi(\alpha_c+1)
\right]
```

Vì `S >= alpha_c` và digamma `\psi` tăng đơn điệu trên miền dương:

```tex
\psi(S+1)\ge \psi(\alpha_c+1)
```

nên mỗi hạng tử trong tổng đều không âm. Do đó:

```tex
u_a\ge 0
```

Ý nghĩa: `u_e` và `u_a` là các tín hiệu uncertainty hợp lệ về mặt toán học. Bài viết vẫn diễn đạt thận trọng rằng chúng là uncertainty proxies, không khẳng định là Bayesian epistemic/aleatoric uncertainty tuyệt đối.

## 4. Proposition: 2:4 Feasibility Preservation

Phần này chứng minh rằng mask luôn thỏa NVIDIA 2:4 structured sparsity.

Sau khi flatten weight tensor, mỗi row được chia thành các block liên tiếp gồm 4 weight. Với mỗi block, ta chọn đúng 2 chỉ số có latent score lớn nhất:

```tex
I_{i,k}^{(l)}
=
\operatorname{Top2}_{j\in\{0,1,2,3\}}
V_{\text{flat},i,4k+j}^{(l)}
```

Mask được định nghĩa:

```tex
M_{\text{flat},i,4k+j}^{(l)}
=
\begin{cases}
1 & \text{if } j\in I_{i,k}^{(l)},\\
0 & \text{otherwise.}
\end{cases}
```

Vì `I_{i,k}^{(l)}` luôn có đúng 2 phần tử, mỗi block 4 phần tử luôn có đúng 2 active weights và 2 inactive weights.

Do đó:

```tex
\text{active density}=50\%
```

trên padded flattened representation.

Ý nghĩa: mỗi lần refresh mask, constraint 2:4 vẫn được bảo toàn. Đây là bảo đảm cấu trúc quan trọng, không phụ thuộc vào giá trị cụ thể của latent score.

## 5. Proposition: Masked-Gradient Isolation

Forward pass dùng effective weight:

```tex
\bm{W}_{\text{eff}}=\bm{W}\odot\bm{M}
```

Khi mask cố định trong một mini-batch, với loss `L`, theo chain rule:

```tex
\frac{\partial L}{\partial W_{ij}}
=
\frac{\partial L}{\partial W_{\text{eff},ij}}
\frac{\partial W_{\text{eff},ij}}{\partial W_{ij}}
```

Mà:

```tex
W_{\text{eff},ij}=W_{ij}M_{ij}
```

nên:

```tex
\frac{\partial W_{\text{eff},ij}}{\partial W_{ij}} = M_{ij}
```

Do đó:

```tex
\nabla_{\bm{W}}L
=
\nabla_{\bm{W}_{\text{eff}}}L\odot\bm{M}
```

Nếu:

```tex
M_{ij}=0
```

thì:

```tex
\frac{\partial L}{\partial W_{ij}}=0
```

Ý nghĩa: dormant weights không nhận task gradient tức thời. Vì vậy regrowth không thể dựa vào ordinary masked backpropagation. Đây là lý do cần ghost-gradient memory hoặc structural memory riêng.

## 6. Ghost-gradient memory

Để dormant links vẫn có đường quay lại, bài viết định nghĩa EMA memory:

```tex
\bar{G}_{ij}^{(t)}
=
\beta_g\bar{G}_{ij}^{(t-1)}
+
(1-\beta_g)\mathbb{I}[M_{ij}^{(t)}=1]G_{ij}^{(t)}
```

Ý nghĩa:

- Nếu link đang active, nó cập nhật historical growth signal.
- Nếu link bị prune, nó không có gradient mới, nhưng vẫn giữ memory từ quá khứ.
- Đây là approximation, không phải exact dense gradient.

Cách viết này giúp bài tránh claim quá mức về dense-gradient recovery, nhưng vẫn có cơ sở kỹ thuật cho dormant regrowth.

## 7. Proposition: Signed First-Order Pruning Criterion

Đây là phần lý thuyết quan trọng nhất để biện minh cho công thức `C_ij`.

Ta định nghĩa evidential risk ratio:

```tex
R=\frac{u_a}{u_e+\epsilon}
```

Nếu prune một connection, tức đưa:

```tex
w_{ij}\to 0
```

thì thay đổi của risk theo Taylor bậc nhất là:

```tex
\Delta R
\approx
R(w_{ij}=0)-R(w_{ij})
\approx
-w_{ij}\frac{\partial R}{\partial w_{ij}}
```

Muốn prune có lợi thì cần:

```tex
\Delta R < 0
```

tức là:

```tex
-w_{ij}\frac{\partial R}{\partial w_{ij}} < 0
```

suy ra:

```tex
w_{ij}\frac{\partial R}{\partial w_{ij}} > 0
```

Vì vậy pruning score chỉ giữ phần dương:

```tex
C_{ij}
=
\operatorname{Rank}
\left(
\left[
w_{ij}\frac{\partial R}{\partial w_{ij}}
\right]_+
\right)
```

Diễn giải:

- Nếu `w_ij * dR/dw_ij > 0`: bỏ connection dự kiến làm giảm risk. Đây là candidate tốt để prune.
- Nếu `w_ij * dR/dw_ij < 0`: bỏ connection dự kiến làm tăng risk. Đây là connection hữu ích, không nên prune.
- Nếu dùng trị tuyệt đối, ta sẽ prune cả những connection nhạy nhưng tốt.

Ý nghĩa: đây là first-order Taylor pruning criterion, có cơ sở từ saliency pruning, nhưng vẫn nên gọi là local first-order heuristic, không phải global optimal pruning theorem.

## 8. Proposition: Local Regrowth Saliency

Regrowth objective:

```tex
L_{\text{grow}}
=
\omega_y u_e
\operatorname{KL}
(\operatorname{Dir}(\bm{\alpha})\Vert\operatorname{Dir}(\mathbf{1}))
```

Mục tiêu này chỉ tăng áp lực regrow khi:

- sample thuộc class quan trọng/hiếm thông qua `\omega_y`,
- model còn thiếu evidence thông qua `u_e`,
- Dirichlet distribution còn gần trạng thái uniform ignorance.

Regrowth score:

```tex
G_{ij}
=
\operatorname{Rank}
\left(
\left|
\frac{\partial L_{\text{grow}}}{\partial w_{ij}}
\right|
\right)
```

Cơ sở toán học:

Gọi:

```tex
Q(\bm{W})=L_{\text{grow}}(\bm{W})
```

Xét perturbation nhỏ:

```tex
w_{ij}\to w_{ij}+\delta_{ij}
```

Taylor bậc nhất:

```tex
Q(w_{ij}+\delta_{ij})-Q(w_{ij})
=
\delta_{ij}
\frac{\partial Q}{\partial w_{ij}}
+ o(|\delta_{ij}|)
```

Nếu ràng buộc:

```tex
|\delta_{ij}|\le \eta
```

thì thay đổi lớn nhất theo bậc nhất là:

```tex
\eta
\left|
\frac{\partial Q}{\partial w_{ij}}
\right|
```

Do đó ranking theo trị tuyệt đối của gradient chọn những connection có khả năng cục bộ lớn nhất để thay đổi objective.

Ý nghĩa: regrower là một local saliency rule. Nó không chứng minh topology tìm được là tối ưu toàn cục. Đây là cách diễn đạt an toàn và khoa học.

## 9. Anti-crystallization

Nếu một layer bị đóng băng cấu trúc, tức growth signal gần như mất:

```tex
\max_{ij}G_{ij}^{(l)}
<
\kappa
\operatorname{EMA}
\left(
\max_{ij}G_{ij}^{(l)}
\right)
```

thì thêm nhiễu:

```tex
\tilde{G}_{ij}^{(l)}
=
G_{ij}^{(l)}
+
\xi_{ij}\sigma(\bm{V}^{(l)})
```

trong đó:

```tex
\xi_{ij}\sim\mathcal{N}(0,1)
```

Ý nghĩa: đây là stochastic exploration term để tránh topology bị crystallize. Không nên gọi nó là SGLD hoặc Langevin sampler nếu không chứng minh phân phối dừng. Trong bản hiện tại, nó được trình bày như heuristic có kiểm soát.

## 10. Proposition: Bounded Morphological State

Latent score update:

```tex
\Delta V_{ij}^{(t)}
=
\mathbb{I}[M_{ij}^{(t)}=1]
(\tilde{G}_{ij}^{(t)}-C_{ij}^{(t)})
+
\mathbb{I}[M_{ij}^{(t)}=0]
\rho\bar{G}_{ij}^{(t-1)}
```

Momentum:

```tex
\mu_{ij}^{(t)}
=
\beta_m\mu_{ij}^{(t-1)}
+
(1-\beta_m)\Delta V_{ij}^{(t)}
```

Sau đó update và clamp:

```tex
V_{ij}^{(t)}
\leftarrow
\operatorname{clamp}
\left(
V_{ij}^{(t-1)}
+
\eta_{\text{struct}}\mu_{ij}^{(t)}
-
\operatorname{mean}(\bm{V}^{(t)}),
-V_{\max},V_{\max}
\right)
```

Vì bước cuối là clamp, nên luôn có:

```tex
V_{ij}^{(t)}\in[-V_{\max},V_{\max}]
```

Ý nghĩa:

- latent topology score không thể nổ vô hạn,
- mask refresh luôn có input hữu hạn,
- kết hợp với proposition 2:4, topology luôn feasible.

Đây không phải chứng minh hội tụ, nhưng là boundedness guarantee.

## 11. Proposition: Controlled Imbalance Modulation

EFL:

```tex
L_{\text{EFL}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\omega_n
\mathcal{F}_n(t)
\left[
L_{\text{CE}}^{(n)}
+
\lambda_{\text{KL}}\mu(t)
\Lambda_{\text{asym}}^{(n)}
L_{\text{KL}}^{(n)}
\right]
```

Focal term:

```tex
\mathcal{F}_n(t)
=
\left(
1-\operatorname{sg}
[\hat{p}_{y_n}^{(n)}]
\right)^{\gamma(t)}
```

Do dùng stop-gradient:

```tex
\nabla_{\bm{e}}\mathcal{F}_n(t)=0
```

Vì vậy focal term chỉ reweight sample, không tạo thêm derivative path qua confidence.

Asymmetric multiplier:

```tex
\Lambda_{\text{asym}}^{(n)}
=
\min(\omega_{y_n},\Lambda_{\max})
```

nên:

```tex
\Lambda_{\text{asym}}^{(n)}\le \Lambda_{\max}
```

Ý nghĩa:

- focal term giúp tập trung vào hard examples mà không bóp méo trực tiếp Dirichlet evidence gradient,
- asymmetric KL được cap để tránh rare-class gradient explosion,
- loss phù hợp hơn với extreme imbalance.

## 12. Tình trạng hiện tại của main body

Sau thay đổi, main body giữ lại:

- định nghĩa Dirichlet/evidence,
- các uncertainty proxies,
- các proposition chính,
- công thức pruning/regrowth/update/loss,
- câu chỉ dẫn rằng proof nằm ở Appendix A.

Main body không còn các đoạn chứng minh dài. Điều này giúp bản thảo gọn hơn và phù hợp hơn với giới hạn 7 trang của AAAI.

## 13. Những điểm nên lưu ý khi viết claim

Các câu nên dùng:

```text
The pruning rule is justified by a signed first-order Taylor criterion.
```

```text
The regrowth rule is a local first-order saliency heuristic.
```

```text
The uncertainty quantities are valid Dirichlet-derived proxies used for structural adaptation.
```

Các câu nên tránh:

```text
The method finds the globally optimal sparse topology.
```

```text
u_e is exact Bayesian epistemic uncertainty.
```

```text
Anti-crystallization is a Langevin sampler.
```

Nói ngắn gọn: phần lý thuyết hiện tại đủ chắc cho một method paper nếu được trình bày là một hệ thống principled, first-order, uncertainty-guided heuristic với các bảo đảm về tính hợp lệ, feasibility, boundedness và gradient isolation.
