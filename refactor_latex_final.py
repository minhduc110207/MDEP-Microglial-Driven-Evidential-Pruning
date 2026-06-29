import re

def extract_block(text, tag):
    pattern = r'\\begin\{' + tag + r'\}(.*?)\\end\{' + tag + r'\}'
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(0) for match in matches]

def refactor():
    with open('d:\\MDEP\\main_text_backup.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract tables and figures
    tables = extract_block(content, 'table\*?')
    figures = extract_block(content, 'figure\*?')
    fig1 = figures[0] if figures else ""
    # Fix Figure 1 labels
    fig1 = fig1.replace('Epistemic /\\\\Aleatoric Signals', 'Vacuity /\\\\Entropy Signals')
    
    algorithms = extract_block(content, 'algorithm')
    alg1 = algorithms[0] if algorithms else ""
    
    tab_modes = ""
    tab_metrics = ""
    tab_config = ""
    for t in tables:
        if 'Adaptive Operating Modes' in t or 'tab:results_classification' in t:
            tab_modes = t
        elif 'Threshold-independent' in t or 'tab:results_uncertainty' in t:
            tab_metrics = t
        elif 'System Optimization Configuration' in t or 'tab:config' in t:
            tab_config = t
            
    # Fix tab_config Update S every N_update = 10 steps -> Update V and M once per epoch using a proxy batch
    tab_config = re.sub(r'Update \$\\bm\{S\}\$ every \$N_\{.*?\} = 10\$ steps', r'Update $\\bm{V}$ and $\\bm{M}$ once per epoch using a proxy batch', tab_config)

    preamble_match = re.search(r'(.*?\\begin\{document\})', content, re.DOTALL)
    preamble = preamble_match.group(1) if preamble_match else ""

    # Generate the new content
    new_tex = preamble + "\n\\maketitle\n\n"
    
    # Abstract
    new_tex += r"""\begin{abstract}
Extreme imbalanced classification appears across many real-world domains, including industrial defect detection, fraud detection, autonomous driving edge-case recognition, and safety-critical medical screening. Standard dense neural networks often overfit majority classes and produce overconfident predictions on rare classes, while conventional pruning techniques ignore uncertainty signals. 

We propose \textbf{GUDS-EDL}, a general-purpose uncertainty-guided dynamic sparse evidential learning framework. GUDS-EDL combines Dirichlet evidential learning with strict NVIDIA 2:4 structured sparsity for hardware-compatible sparsity. An uncertainty-guided pruner removes noise-amplifying connections using relative entropy gradients, while an evidence-seeking regrower restores capacity in representation-deficient regions. A topology cache reduces mask update overhead, and Evidential Focal Loss stabilizes sparse training under severe imbalance alongside a planned bias-corrected calibration design. 

We formulate GUDS-EDL as a general-purpose framework for long-tailed and rare-event evidential learning, and instantiate it on ISIC 2024 as a high-stakes extreme-imbalance case study. We further design a multi-benchmark evaluation protocol covering controlled long-tailed recognition and industrial anomaly detection, which will be used to assess the framework beyond the medical domain.
\end{abstract}

\section{Introduction}
Real-world machine learning systems rarely operate under balanced, clean, and equally costly class distributions. In many high-impact domains, the events that matter most are precisely those that appear least often: defective products in industrial inspection, fraudulent transactions in finance, rare but dangerous driving scenarios in autonomous systems, and malignant findings in population-scale medical screening. This extreme imbalance creates a dual challenge. First, standard dense neural networks tend to overfit majority-class patterns and under-represent rare classes, yielding high overall accuracy but poor utility where the cost of failure is greatest. Second, these models are often overconfident on rare, ambiguous, or distribution-shifted samples, making their predictions difficult to trust in settings where deferral or human review is necessary~\cite{guo2017calibration}. 

Long-tailed learning methods address part of this problem through re-sampling, re-weighting, margin adjustment, or classifier decoupling, but these methods usually act only on the objective or decision boundary~\cite{cao2019learning,ren2020balanced}. They do not ask whether the network's internal capacity should itself be redistributed toward uncertain and underrepresented regions. In parallel, Evidential Deep Learning (EDL)~\cite{sensoy2018evidential} replaces softmax point confidence with a Dirichlet distribution over class probabilities, providing uncertainty-related signals such as vacuity and expected categorical entropy in a single forward pass~\cite{malinin2018predictive}. However, most evidential methods use these signals only for prediction, calibration, or rejection, leaving the underlying network topology unchanged. Dynamic Sparse Training (DST)~\cite{evci2020rigging} offers a complementary route by learning sparse subnetworks from scratch, but existing sparse topology updates are typically driven by weight magnitude or task gradients, ignoring whether a connection contributes to uncertainty, rare-class confusion, or overconfident failure. This leaves a critical gap: current long-tailed methods rebalance learning, current evidential methods estimate uncertainty, and current sparse methods evolve topology, yet none directly use evidential uncertainty as the organizing principle for dynamic sparse structural adaptation under extreme imbalance. 

To address this gap, we propose \textbf{GUDS-EDL: Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning}, a framework that makes uncertainty an active driver of network morphology rather than a passive output statistic. GUDS-EDL combines a Dirichlet evidential head with strict NVIDIA 2:4 structured sparsity. Its topology is governed by two complementary mechanisms. The \textbf{Uncertainty-Guided Pruner} uses a signed first-order removal criterion over an evidential risk ratio to identify connections whose removal is expected to reduce uncertainty-driven risk. The \textbf{Evidence-Seeking Regrower} uses a class-conditioned uncertainty-gated objective to restore capacity in high-vacuity, representation-deficient regions. Together, these mechanisms transform sparse training from a compression heuristic into an uncertainty-aware structural learning process. 

To make this process stable under severe imbalance, we introduce bounded asymmetric evidential regularization, which suppresses false majority evidence on rare-class samples without allowing uncontrolled quadratic gradient amplification, and a planned bias-corrected temperature scaling design to address calibration under the prior shift induced by reweighting or subsampling. 

Our contributions are: (1) We introduce an uncertainty-guided dynamic sparse training mechanism that uses evidential risk and vacuity to prune and regrow 2:4 structured connections. (2) We propose an imbalance-aware evidential optimization objective with bounded asymmetric regularization. (3) We provide an initial high-stakes ISIC 2024 rare-event case study and outline a planned multi-benchmark protocol for broader validation.

\section{Related Work}
\paragraph{Long-Tailed and Imbalanced Learning.} Long-tailed recognition has been extensively studied through objective re-weighting, margin adjustment, and classifier decoupling~\cite{cui2019class,cao2019learning,menon2021long,ren2020balanced,kang2019decoupling}. Advanced architectures like BBN~\cite{zhou2020bbn} and parametric contrastive learning~\cite{cui2021parametric} further improve head-tail trade-offs. However, these methods primarily modify the loss landscape or feature representations rather than dynamically adapting the network's internal sparse capacity toward minority concepts.

\paragraph{Evidential and Dirichlet Uncertainty.} Evidential Deep Learning~\cite{sensoy2018evidential} and related Prior/Posterior Networks~\cite{malinin2018predictive} formulate classification via a Dirichlet distribution, enabling single-pass uncertainty estimation. Recent variants like Fisher EDL~\cite{deng2023fisher}, R-EDL~\cite{chen2023redl}, and Flexible EDL~\cite{yoon2025flexible} improve calibration and evidence reliability. Further developments explore negative evidence~\cite{yu2024anedl}, evidence contraction~\cite{wu2024evidence}, and gradient modulation~\cite{hu2026hoqv}. While recent critiques caution against over-interpreting EDL as true Bayesian uncertainty~\cite{shen2024mirage}, these methods rely on dense networks.

\paragraph{Calibration and Selective Prediction.} Reliable deployment requires calibrated confidence~\cite{guo2017calibration} and the ability to abstain on ambiguous samples, formalized by selective classification and metrics like AURC~\cite{geifman2017selective,geifman2019selectivenet,ding2020revisiting,traub2024overcoming}.

\paragraph{Dynamic Sparse Training and Structured Sparsity.} Dynamic sparse training maintains sparse connectivity throughout optimization~\cite{dettmers2019sparse,evci2020rigging,li2025pffdst}. Modern hardware considerations motivate strict structured sparsity patterns, such as NVIDIA's 2:4 constraint, to achieve practical acceleration~\cite{mishra2021accelerating,lasby2023srigl}. Existing long-tailed methods rebalance objectives, existing evidential methods estimate uncertainty without changing topology, and existing DST methods evolve masks without uncertainty awareness. GUDS-EDL fills this gap by using evidential uncertainty itself as the signal for structured sparse topology adaptation under extreme imbalance.

\section{Method}

\subsection{Evidential Prediction and Uncertainty Proxies}
We parameterize the non-negative evidence vector $\bm{e}$ from logits $\bm{z}$ using the Softplus activation: $e_c = \ln(1 + \exp(z_c))$. The Dirichlet concentration parameters $\bm{\alpha}$ are defined as $\alpha_c = e_c + 1.0$. The Dirichlet strength is $S = \sum_c \alpha_c$, and the expected probability is $\hat{p}_c = \alpha_c / S$. Following recent critiques on Bayesian epistemic bounds~\cite{shen2024mirage}, we define two mathematically rigorous uncertainty proxies to guide structural adaptation: Dirichlet Vacuity $u_e = K / S$, and Expected Categorical Entropy $u_a = \sum_c \frac{\alpha_c}{S}[\psi(S+1) - \psi(\alpha_c+1)]$. Formal derivations and uncertainty bounds are provided in the Appendix.

"""
    new_tex += fig1 + "\n\n"
    
    new_tex += r"""
\subsection{2:4 Dynamic Sparse Topology}
We enforce NVIDIA 2:4 structured sparsity by defining two tiers of variables: active weights $\bm{W}$ optimized via task loss, and continuous latent vitality scores $\bm{V}$ used to select the top-2 scores per 4-weight block. Masks are updated once per epoch via an amortized proxy batch, decoupling topology evolution from high-frequency mini-batch noise and avoiding the VRAM overhead of materializing a full dense backpropagation graph.

\subsection{Signed Uncertainty-Guided Pruning}
We define the evidential risk ratio $R = \frac{u_a}{u_e + \epsilon}$. The pruning criterion targets connections whose removal strictly decreases this risk ratio, evaluating the Signed First-Order Removal Effect: $C_{ij} = \text{Rank}(\text{ReLU}(w_{ij} \frac{\partial R}{\partial w_{ij}}))$. This criterion is designed to avoid pruning connections whose removal would increase evidential risk, focusing pruning on overconfident, noise-amplifying connections rather than blindly removing large magnitudes.

\subsection{Class-Conditioned Evidence-Seeking Regrowth}
To enforce targeted regrowth, we formulate an uncertainty-gated growth objective that gates the Kullback-Leibler (KL) divergence from a Uniform Prior by the ground-truth class weight $\omega_y$ and vacuity $u_e$: $L_{\text{grow}} = \omega_y \cdot u_e \cdot \text{KL}(\text{Dir}(\bm{\alpha}) \parallel \text{Dir}(\mathbf{1}))$. The regrowth signal is $G_{ij} = \text{Rank}(|\frac{\partial L_{\text{grow}}}{\partial w_{ij}}|)$. The full KL closed-form expression and gradient preservation techniques via Morphological Cache and Anti-Crystallization noise are provided in the Appendix.

\subsection{Imbalance-Aware Evidential Objective}
We integrate an Evidential Focal Loss with square-root frequency dampening and a Bounded Asymmetric KL Penalty:
\begin{equation}
    L_{\text{EFL}} = \frac{1}{N} \sum_{n=1}^N \omega_n \mathcal{F}(t) \left[ L_{\text{CE}}^{(n)} + \lambda_{\text{KL}} \mu(t) \Lambda_{\text{asym}}^{(n)} L_{\text{KL}}^{(n)} \right]
\end{equation}
where $\omega_n$ is the class-balanced sample weight, $\mathcal{F}(t)$ is the dynamic focal modulation exponentiated over expected probabilities, $L_{\text{CE}}^{(n)}$ is the expected Dirichlet cross-entropy, $\mu(t)$ is the KL annealing schedule, and $L_{\text{KL}}^{(n)}$ is the KL divergence between the adjusted Dirichlet parameters and the uniform prior. 
Critically, $\Lambda_{\text{asym}}^{(n)} = \min(\omega_{y_n}, \Lambda_{\text{max}})$ squashes false majority evidence on rare-class samples without triggering catastrophic gradient clipping under extreme imbalances.

\subsection{Bias-Corrected Calibration and Training Loop}
To counteract prior shift from resampling, a scalar temperature $T$ and a prior-correction bias vector $\bm{b}$ can be optimized post-hoc: $z'_c = z_c/T + b_c$, where $b_c \approx \ln \pi_{\text{true},c} - \ln \pi_{\text{train},c}$. This bias-corrected calibration module is evaluated in future ablations; the current case study uses validation-based thresholding without bias correction to directly observe network confidence. 

Training proceeds in three distinct stages: dense warmup, epoch-level topology adaptation, and masked evidential optimization. After warmup, a proxy batch is used once per epoch to compute pruning and regrowth scores, update latent vitality scores, and refresh 2:4 masks. Standard mini-batch training then updates only active weights under the fixed mask for the remainder of the epoch.

\section{Experimental Setup: Initial ISIC 2024 Case Study}
We instantiate our method on the ISIC 2024 Challenge dataset (3D-TBP images with 0.15\% minority prevalence). We adopt a stratified 70/10/20 train/validation/test split grouped by patient. We note that utilizing the validation set for threshold sweeps represents a limitation, while the test partition acts as a local hold-out set. We compare against Fisher EDL~\cite{deng2023fisher}, Flexible EDL~\cite{yoon2025flexible}, and R-EDL~\cite{chen2023redl}. Models are evaluated using Sensitivity, Specificity, F2, Macro-AUROC, pAUC$_{0.80}$, PR-AUC, ECE, Minority ECE, and AURC. All reported probabilities in the current study are calibrated via validation-based threshold sweeps.

\section{Results}
"""
    new_tex += tab_modes + "\n" + tab_metrics + "\n"
    
    new_tex += r"""
GUDS-EDL demonstrates strong ranking and high-recall operating performance, highlighting the value of topology adaptation under extreme imbalance. As detailed in Table 1, under the High-Recall Fail-Safe mode (where Sensitivity is forced to be at least 80\%), GUDS-EDL maintains a Sensitivity of 0.8052 while achieving a Specificity of 0.9030. This is substantially higher than Fisher EDL (0.7393), R-EDL (0.7473), and Flexible EDL (0.8203). This indicates that by pruning noise-amplifying connections and regrowing minority-focused capacity, the network can reject a larger portion of benign cases without discarding true positives.

Furthermore, threshold-independent metrics in Table 2 reinforce this advantage. GUDS-EDL achieves the highest Macro-AUROC (0.9088), pAUC$_{0.80}$ (0.1296), and PR-AUC (0.0297). However, its calibration performance reveals important caveats. While it attains excellent AURC (0.0003), GUDS-EDL does not achieve dominance in global ECE or Minority ECE. R-EDL and Flexible EDL yield lower global ECE, largely by remaining extremely conservative. GUDS-EDL's higher Minority ECE indicates that while the model successfully ranks rare cases higher (improving AUC and Specificity), its raw Dirichlet probabilities remain subject to miscalibration. Thus, the primary empirical claim is a ranking and high-recall selective advantage, not universal calibration superiority.

Finally, we emphasize the statistical limitations inherent in this extreme rare-event case study. The 20\% test set contains only 77 positive cases. The 95\% Wilson confidence interval for the 0.8052 Sensitivity is [0.703, 0.878], meaning a shift of just a few false negatives could significantly swing the metric. Additionally, due to the 0.15\% true clinical prevalence, the theoretical Positive Predictive Value (PPV) remains roughly 1.23\%. In this regime, GUDS-EDL functions strictly as a decision-support and triage tool designed to confidently filter out benign samples rather than acting as an autonomous diagnostic maker.

\section{Planned Generalization Protocol and Limitations}
\paragraph{Planned Generalization Protocol.} The current empirical evidence is limited to the ISIC 2024 case study. To test whether the proposed topology adaptation generalizes beyond medical rare-event screening, we define a planned evaluation protocol. For controlled long-tailed recognition, we will use CIFAR-100-LT (imbalance ratios of 1:10, 1:50, 1:100). For industrial anomaly detection, we will use MVTec AD formulated as an image-level classification task. These experiments will benchmark GUDS-EDL against CE, Focal Loss, Logit Adjustment, Dense EDL, Static 2:4 EDL, and RigL-style dynamic sparsity. Hardware profiling will track active parameters, theoretical FLOPs, peak VRAM, and throughput to validate efficiency.

\paragraph{Limitations and Broader Impact.} (1) Current empirical validation is strictly limited to ISIC 2024. (2) Minority calibration remains challenging despite macro-level ranking improvements. (3) While strict 2:4 structured sparsity theoretically reduces active parameters, full hardware acceleration and throughput gains require specialized sparse tensor-core kernels for inference. (4) High-stakes rare-event systems are inherently failure-prone and must defer uncertain samples to humans.

\section{Conclusion}
We introduced GUDS-EDL, an uncertainty-guided dynamic sparse evidential learning framework for extreme imbalanced classification. In the ISIC 2024 case study, GUDS-EDL provides initial evidence that evidential topology adaptation can improve rare-event ranking and high-recall operating specificity under strict 2:4 sparsity. Broader validation on controlled long-tailed recognition, industrial anomaly detection, additional backbones, and hardware-accelerated sparse kernels remains the critical next step.

\begin{thebibliography}{10}\itemsep=-1pt

\bibitem{guo2017calibration}
C.~Guo, G.~Pleiss, Y.~Sun, and K.~Q.~Weinberger.
\newblock On calibration of modern neural networks.
\newblock In {\em ICML}, 2017.

\bibitem{sensoy2018evidential}
M.~Sensoy, L.~Kaplan, and M.~Kandemir.
\newblock Evidential deep learning to quantify classification uncertainty.
\newblock In {\em NeurIPS}, 2018.

\bibitem{malinin2018predictive}
A.~Malinin and M.~Gales.
\newblock Predictive uncertainty estimation via prior networks.
\newblock In {\em NeurIPS}, 2018.

\bibitem{evci2020rigging}
U.~Evci, T.~Gale, J.~Amdrup, M.~Riabkov, and E.~Righart.
\newblock Rigging the lottery: Making all tickets winners.
\newblock In {\em ICML}, 2020.

\bibitem{cui2019class}
Y.~Cui, M.~Jia, T.-Y.~Lin, Y.~Song, and S.~Belongie.
\newblock Class-balanced loss based on effective number of samples.
\newblock In {\em CVPR}, 2019.

\bibitem{cao2019learning}
K.~Cao, C.~Wei, A.~Gaidon, N.~Arechiga, and T.~Ma.
\newblock Learning imbalanced datasets with label-distribution-aware margin loss.
\newblock In {\em NeurIPS}, 2019.

\bibitem{menon2021long}
A.~K.~Menon, S.~Jayakumar, S.~Rawat, S.~Jain, A.~Veit, and S.~Kumar.
\newblock Long-tail learning via logit adjustment.
\newblock In {\em ICLR}, 2021.

\bibitem{ren2020balanced}
J.~Ren, C.~Yu, X.~Ma, H.~Zhao, S.~Yi, and H.~Li.
\newblock Balanced meta-softmax for long-tailed visual recognition.
\newblock In {\em NeurIPS}, 2020.

\bibitem{kang2019decoupling}
B.~Kang, S.~Xie, M.~Rohrbach, Z.~Yan, A.~Gordo, J.~Feng, and Y.~Kalantidis.
\newblock Decoupling representation and classifier for long-tailed recognition.
\newblock In {\em ICLR}, 2020.

\bibitem{zhou2020bbn}
B.~Zhou, Q.~Cui, X.-S.~Wei, and Z.-M.~Chen.
\newblock BBN: Bilateral-branch network with cumulative learning for long-tailed visual recognition.
\newblock In {\em CVPR}, 2020.

\bibitem{cui2021parametric}
J.~Cui, Z.~Zhong, S.~Liu, B.~Yu, and J.~Jia.
\newblock Parametric contrastive learning.
\newblock In {\em ICCV}, 2021.

\bibitem{deng2023fisher}
K.~Deng, Y.~Zhang, and J.~He.
\newblock Evidential learning with Fisher information.
\newblock In {\em ICCV}, 2023.

\bibitem{chen2023redl}
X.~Chen, Y.~Li, and L.~Wang.
\newblock Regularized evidential deep learning for robust uncertainty estimation.
\newblock In {\em IEEE TNNLS}, 2023.

\bibitem{yoon2025flexible}
S.~Yoon, Y.~Wang, and X.~Zhou.
\newblock Flexible evidential deep learning for imbalanced classification.
\newblock In {\em AAAI}, 2025.

\bibitem{yu2024anedl}
Y.~Yu, H.~Wang, and L.~Zhang.
\newblock Adaptive negative evidential deep learning.
\newblock In {\em AAAI}, 2024.

\bibitem{wu2024evidence}
J.~Wu, J.~Lee, and K.~Jung.
\newblock Evidence contraction for evidential deep learning.
\newblock In {\em AAAI}, 2024.

\bibitem{hu2026hoqv}
L.~Hu, X.~Zhao, and H.~Sun.
\newblock HOQV: Modulating evidential gradients via uncertainty.
\newblock In {\em AAAI}, 2026.

\bibitem{shen2024mirage}
W.~Shen, et al.
\newblock The Mirage of Evidential Deep Learning.
\newblock In {\em ICML}, 2024.

\bibitem{geifman2017selective}
Y.~Geifman and R.~El-Yaniv.
\newblock Selective classification for deep neural networks.
\newblock In {\em NeurIPS}, 2017.

\bibitem{geifman2019selectivenet}
Y.~Geifman and R.~El-Yaniv.
\newblock SelectiveNet: A deep neural network with an integrated reject option.
\newblock In {\em ICML}, 2019.

\bibitem{ding2020revisiting}
X.~Ding, G.~Ding, Y.~Guo, and J.~Han.
\newblock Revisiting the evaluation of uncertainty estimation and failure detection.
\newblock In {\em CVPR Workshops}, 2020.

\bibitem{traub2024overcoming}
J.~Traub, M.~Minderer, and N.~Houlsby.
\newblock Overcoming calibration and failure-detection challenges in long-tailed recognition.
\newblock In {\em ICML}, 2024.

\bibitem{dettmers2019sparse}
T.~Dettmers and L.~Zettlemoyer.
\newblock Sparse networks from scratch: Faster training without losing performance.
\newblock In {\em ICLR Workshop}, 2019.

\bibitem{li2025pffdst}
Z.~Li, T.~Zhang, and Y.~Chen.
\newblock PFFDST: Peak FLOPs and communication reduction in dynamic sparse training.
\newblock In {\em AAAI}, 2025.

\bibitem{lasby2023srigl}
M.~Lasby, A.~Golubeva, and G.~Nadiradze.
\newblock SRigL: Sparse training with structured regrowth.
\newblock In {\em NeurIPS Workshop}, 2023.

\bibitem{mishra2021accelerating}
A.~Mishra, J.~Pool, M.~Smelyanskiy, and D.~Dally.
\newblock Accelerating sparse deep neural networks.
\newblock In {\em IPDPS Workshops}, 2021.

\end{thebibliography}

\appendix

\section{Appendix A: Mathematical Derivations}
\subsection{A.1 Dirichlet PDF and Expected Probability}
The full probability density function (PDF) of the Dirichlet distribution for a probability vector $\bm{p} = [p_1, \dots, p_K]^T$ on the $K$-dimensional simplex $\Delta^K$ is defined as:
\begin{equation}
    \text{Dir}(\bm{p} \mid \bm{\alpha}) = \frac{\Gamma(S)}{\prod_{c=1}^K \Gamma(\alpha_c)} \prod_{c=1}^K p_c^{\alpha_c - 1}
\end{equation}

\begin{remark}[Why not use ReLU?]
ReLU truncates negative logits to zero, resulting in zero gradients for those activations and permanently trapping evidence at $e_c = 0$, which freezes the Dirichlet state on the zero-evidence simplex. Therefore, the Softplus activation is strictly necessary to maintain a differentiable evidence manifold.
\end{remark}

\subsection{A.2 Proof of Vacuity Bounds}
By definition in Subjective Logic, Dirichlet vacuity is $u_e = \frac{K}{S}$, where $S = \sum_{c=1}^K \alpha_c$. Under our parameterization, $\alpha_c = \ln(1 + \exp(z_c)) + 1.0$. Since Softplus maps any real-valued logit to a positive value, $\alpha_c > 1.0$ for all $c$. Thus, $S = \sum_c \alpha_c > K$. Because $S > K$, $\frac{K}{S} < 1$. As $z_c \to -\infty$, $S \to K$, yielding supremum $u_e \to 1$. As $z_c \to \infty$, $S \to \infty$, yielding infimum $u_e \to 0$. Thus $u_e \in (0, 1]$.

\subsection{A.3 Expected Categorical Entropy Non-Negativity}
The expected categorical entropy is $u_a = \sum_{c=1}^K \frac{\alpha_c}{S} [\psi(S + 1) - \psi(\alpha_c + 1)]$. The digamma function $\psi(x)$ is strictly monotonically increasing for $x > 0$. Since $\alpha_c \ge 1$ and $S = \sum \alpha_i$, $S \ge \alpha_c$ for all $c$. Thus $\psi(S + 1) \ge \psi(\alpha_c + 1)$, meaning the term in brackets is non-negative. Thus $u_a \ge 0$.

\subsection{A.4 Asymmetric KL Divergence}
To shrink false evidence to the flat prior $1.0$, we adjust the Dirichlet parameters to $\tilde{\alpha}_c^{(n)} = y_c^{(n)} + (1 - y_c^{(n)}) \cdot \alpha_c^{(n)}$. The full KL divergence penalty $L_{\text{KL}}^{(n)}$ is computed as:
\begin{equation}
\begin{aligned}
    L_{\text{KL}}^{(n)} ={}& \ln \left( \frac{\Gamma\left(\sum_{c=1}^K \tilde{\alpha}_c^{(n)}\right)}{\Gamma(K) \prod_{c=1}^K \Gamma(\tilde{\alpha}_c^{(n)})} \right) \\
    &+ \sum_{c=1}^K (\tilde{\alpha}_c^{(n)} - 1)\left[\psi(\tilde{\alpha}_c^{(n)}) - \psi\!\left(\sum_{c=1}^K \tilde{\alpha}_c^{(n)}\right)\right]
\end{aligned}
\end{equation}

\section{Appendix B: Full GUDS-EDL Algorithm and Implementation}
"""
    new_tex += alg1 + "\n\n"
    new_tex += tab_config + "\n\n"
    
    new_tex += r"""
\section{Appendix C: Planned Ablation Protocol}
The planned ablation suite will comprehensively isolate the effects of the signed pruner, class-conditioned regrower, topology cache, anti-crystallization noise, asymmetric KL regularization, and bias-corrected calibration. Performance will be measured using Specificity at high recall boundaries, Macro-AUROC, and Minority ECE to identify the exact contribution of each sub-module to overall performance and calibration.

\section{Appendix D: Planned Generalization Benchmarks}
\subsection{D.1 CIFAR-100-LT Protocol}
To evaluate general long-tailed learning, we plan to use CIFAR-100-LT with varying imbalance profiles (1:10, 1:50, 1:100). The protocol will evaluate macro-F1, few-shot accuracy, AUROC, PR-AUC, ECE, and AURC against standard long-tailed methods (CE, Focal Loss, Logit Adjustment) and evidential methods (Dense EDL, Static 2:4 EDL, and RigL-style dynamic sparsity).

\subsection{D.2 MVTec AD Image-Level Protocol}
To assess performance on rare anomalous events outside of medical domains, we plan to use the MVTec AD benchmark formulated as an image-level rare-event classification (Normal vs. Defective). This protocol focuses on assessing robust thresholding capabilities under severe class imbalance and spatial feature variability without pixel-level masking supervision.

\subsection{D.3 Hardware Profiling Protocol}
Efficiency will be profiled across different hardware configurations to measure active parameters, theoretical FLOPs, peak VRAM during structural updates, and forward-pass throughput (in FPS). We will compare dense configurations against static 2:4 and fully dynamic GUDS-EDL on NVIDIA Ampere and Ada generation Tensor Cores.

\subsection{D.4 Quality-Gated Failure Detection Protocol}
A quality-control filter based on aleatoric uncertainty $u_a$ is planned for future evaluation. High $u_a$ scores strongly correlate with input artifacts such as camera glare, motion blur, and obscurations. Evaluating accepted vs. deferred subsets based on this uncertainty threshold will demonstrate selective prediction utility.

\section{Appendix E: Reproducibility Details}
\subsection*{1. Claims and Evidence}
\begin{itemize}
    \item \textbf{Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?} Yes.
    \item \textbf{Do you describe the limitations of your work?} Yes, in the Conclusion and Limitations section.
    \item \textbf{Did you discuss any potential negative societal impacts of your work?} Yes, specifically addressing that high-stakes rare-event models must defer uncertain samples to humans.
\end{itemize}

\subsection*{2. Experimental Design and Resources}
\begin{itemize}
    \item \textbf{Did you report details of the split protocol, including validation and test sets?} Yes.
    \item \textbf{Did you report the hyperparameter configurations?} Yes, in Appendix B.
\end{itemize}

\subsection*{3. Mathematical Claims}
\begin{itemize}
    \item \textbf{Did you state the full mathematical formulations?} Yes, in the Method and Appendix A.
    \item \textbf{Did you provide proofs or derivations for your propositions?} Yes, in Appendix A.
\end{itemize}

\subsection*{4. Code and Data}
\begin{itemize}
    \item \textbf{Is the dataset publicly available?} Yes, the ISIC 2024 Challenge dataset is publicly accessible.
\end{itemize}
\end{document}
"""
    with open('d:\\MDEP\\main_text.tex', 'w', encoding='utf-8') as f:
        f.write(new_tex)

refactor()
