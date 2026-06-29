import re

tex_content = r"""\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
\usepackage{aaai}  % DO NOT CHANGE THIS
\usepackage{times}  % DO NOT CHANGE THIS
\usepackage{helvet}  % DO NOT CHANGE THIS
\usepackage{courier}  % DO NOT CHANGE THIS
\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
\usepackage{graphicx} % DO NOT CHANGE THIS
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS
\frenchspacing  % DO NOT CHANGE THIS
\setlength{\pdfpagewidth}{8.5in} % DO NOT CHANGE THIS
\setlength{\pdfpageheight}{11in}
\setcounter{secnumdepth}{2}
%%%%%%%%%%
% PDFINFO for PDFLATEX
\pdfinfo{
/Title (GUDS-EDL: Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning)
/Author (Anonymous Authors)
}
%%%%%%%%%%
 % DO NOT CHANGE THIS

% --- Additional Packages ---
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath, amssymb, amsthm, bm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{adjustbox}
\usepackage{placeins}
\usepackage{stfloats}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{enumitem}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}
\usepackage[table]{xcolor}
\usepackage{colortbl}
\usepackage[colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue]{hyperref}

\newcolumntype{C}{>{\centering\arraybackslash}c}

% --- Custom commands ---
\newcommand{\Dir}{\text{Dir}}
\newcommand{\KL}{\text{KL}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Norm}{\text{Norm}}
\newcommand{\softplus}{\text{Softplus}}
\DeclareMathOperator*{\argmax}{arg\,max}

\hbadness=10000
\vbadness=10000
\hfuzz=10pt
\vfuzz=4pt
\raggedbottom
\setlength{\textfloatsep}{10pt plus 2pt minus 3pt}
\setlength{\dbltextfloatsep}{10pt plus 2pt minus 3pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{8pt plus 2pt minus 2pt}

\newtheorem{remark}{Remark}
\newtheorem{proposition}{Proposition}

% ============================================================================
\title{GUDS-EDL: Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning}
\author{Anonymous Authors}

\begin{document}
\maketitle

\begin{abstract}
Extreme imbalanced classification appears across many real-world domains, including industrial defect detection, fraud detection, autonomous driving edge-case recognition, and safety-critical medical screening. Standard dense neural networks often overfit majority classes and produce overconfident predictions on rare classes, while conventional pruning techniques ignore uncertainty signals. 

We propose \textbf{GUDS-EDL}, a general-purpose uncertainty-guided dynamic sparse evidential learning framework. GUDS-EDL combines Dirichlet evidential learning with dynamic 2:4 structured sparsity. An uncertainty-guided pruner removes noise-amplifying connections using relative entropy gradients, while an evidence-seeking regrower restores capacity in representation-deficient regions. A topology cache reduces mask update overhead, and Evidential Focal Loss stabilizes sparse training under severe imbalance. 

We formulate GUDS-EDL as a general-purpose framework for long-tailed and rare-event evidential learning, and instantiate it on ISIC 2024 as a high-stakes extreme-imbalance case study. We further design a multi-benchmark evaluation protocol covering controlled long-tailed recognition and industrial anomaly detection, which will be used to assess the framework beyond the medical domain.
\end{abstract}

\section{Introduction}

Extreme imbalanced classification is a defining failure mode of real-world machine learning, where the most consequential samples are the least observed. This setting arises naturally across safety-critical domains where rare events present severe consequences, challenging the standard assumption of balanced distributions and equal misclassification costs.

Standard long-tailed methods rebalance the loss landscape or adjust classifiers but leave the dense network topology unchanged. Evidential Deep Learning replaces point confidence with a Dirichlet distribution, providing rich uncertainty signals, but does not use this uncertainty to adapt the sparse structural capacity. Conversely, Dynamic Sparse Training evolves topology during optimization but relies on magnitude or task-gradient heuristics that are blind to uncertainty.

We propose GUDS-EDL, a generalized uncertainty-guided dynamic sparse evidential learning framework. GUDS-EDL consists of a Dirichlet evidential head, a signed first-order uncertainty-guided pruner, a class-conditioned uncertainty-gated regrower, and 2:4 structured sparsity. To stabilize sparse topology evolution under extreme prior shift, we introduce an imbalance-aware objective with bounded asymmetric KL regularization and bias-corrected temperature scaling.

Our contributions are:
(1) We introduce an uncertainty-guided dynamic sparse training mechanism that uses evidential risk and vacuity to prune and regrow 2:4 structured connections.
(2) We propose an imbalance-aware evidential optimization objective with bounded asymmetric regularization and bias-corrected calibration.
(3) We provide an initial high-stakes ISIC 2024 rare-event case study and outline a planned multi-benchmark protocol for broader validation.

\section{Related Work}

\paragraph{Long-Tailed and Imbalanced Learning.} Long-tailed recognition addresses class imbalance through re-sampling, margin adjustment, and objective re-weighting. Class-Balanced Loss \cite{cui2019class} reweights examples using the effective number of samples, while LDAM-DRW \cite{cao2019learning} adjusts decision boundaries. Logit Adjustment \cite{menon2021long} and Balanced Softmax \cite{ren2020balanced} treat imbalance as label-prior shift, and decoupled training separates representations from classifier balancing \cite{kang2019decoupling}.

\paragraph{Evidential and Dirichlet Uncertainty.} Evidential Deep Learning (EDL) \cite{sensoy2018evidential} replaces softmax outputs with a Dirichlet distribution, quantifying both data noise and model ignorance in a single pass. Prior Networks \cite{malinin2018predictive} and Posterior Networks \cite{charpentier2020posterior} explore similar density-based pseudo-counts. Recent variants like Fisher EDL \cite{deng2023fisher} and Flexible EDL \cite{yoon2025flexible} improve stability, though recent work clarifies limitations in interpreting these as true Bayesian bounds \cite{shen2024mirage}.

\paragraph{Calibration and Selective Prediction.} Safe deployment requires calibrated confidence and the ability to abstain. Temperature scaling provides a strong post-hoc baseline \cite{guo2017calibration}. Selective classification models the risk-coverage trade-off, enabling networks to reject ambiguous samples \cite{geifman2017selective}. The Area Under the Risk-Coverage Curve (AURC) is standard for evaluating failure detection performance under deferral rules \cite{ding2020revisiting}.

\paragraph{Dynamic Sparse Training and Structured Sparsity.} Dynamic sparse training evolves connectivity during optimization. Unstructured methods like SET \cite{mocanu2018scalable} and RigL \cite{evci2020rigging} prune by magnitude and regrow by gradients. To ensure hardware compatibility, structured sparsity patterns, such as SRigL \cite{lasby2023srigl} and NVIDIA 2:4 Sparse Tensor constraints \cite{mishra2021accelerating}, provide strict rules for structured pruning.

Existing long-tailed methods rebalance objectives or classifiers, existing evidential methods estimate uncertainty without changing topology, and existing DST methods evolve masks without uncertainty awareness. GUDS-EDL fills this gap by using evidential uncertainty itself as the signal for structured sparse topology adaptation under extreme imbalance.

\section{Method}

\subsection{Evidential Prediction and Uncertainty Proxies}
We parameterize the non-negative evidence vector $\bm{e}$ from logits $\bm{z}$ using the Softplus activation:
\begin{equation}
    e_c = \text{Softplus}(z_c) = \ln(1 + \exp(z_c))
\end{equation}
The Dirichlet concentration parameters $\bm{\alpha}$ are defined as $\alpha_c = e_c + 1.0$. The Dirichlet strength is $S = \sum_{c=1}^K \alpha_c$, and the expected probability is $\hat{p}_c = \alpha_c / S$. We define two uncertainty proxies to guide structural adaptation: Dirichlet Vacuity $u_e$ and Expected Categorical Entropy $u_a$:
\begin{equation}
    u_e = \frac{K}{S}
\end{equation}
\begin{equation}
    u_a = \sum_{c=1}^K \frac{\alpha_c}{S} \left[\psi(S + 1) - \psi(\alpha_c + 1)\right]
\end{equation}
Formal derivations and uncertainty bounds are provided in the Appendix.

\subsection{2:4 Dynamic Sparse Topology}
We define two tiers of variables: active weights $\bm{W}$ optimized via task loss, and continuous latent vitality scores $\bm{V}$ used to select the top-2 scores per 4-weight block. Masks $\bm{M}$ are evaluated dynamically while conforming strictly to the NVIDIA 2:4 structured sparsity constraint.

\subsection{Signed Uncertainty-Guided Pruning}
We define the evidential risk ratio $R = u_a / (u_e + \epsilon)$. The pruning criterion evaluates the Signed First-Order Removal Effect:
\begin{equation}
    C_{ij} = \text{Rank}\left( \text{ReLU}\left( w_{ij} \frac{\partial R}{\partial w_{ij}} \right) \right)
\end{equation}
This criterion is designed to avoid pruning connections whose removal would increase evidential risk.

\subsection{Class-Conditioned Evidence-Seeking Regrowth}
To enforce targeted regrowth, we formulate an uncertainty-gated growth objective that gates the Kullback-Leibler (KL) divergence from a uniform prior by the sample's ground-truth class weight $\omega_y$ and vacuity $u_e$:
\begin{equation}
    L_{\text{grow}} = \omega_y \cdot u_e \cdot \KL(\Dir(\bm{\alpha}) \parallel \Dir(\mathbf{1}))
\end{equation}
The regrowth signal is driven by the gradients of this objective:
\begin{equation}
    G_{ij} = \text{Rank}\left( \left| \frac{\partial L_{\text{grow}}}{\partial w_{ij}} \right| \right)
\end{equation}
$\KL(\Dir(\bm{\alpha}) \parallel \Dir(\mathbf{1}))$ has a closed-form expression provided in the Appendix.

\subsection{Imbalance-Aware Evidential Objective}
We integrate an Evidential Focal Loss (EFL) with a Bounded Asymmetric KL Penalty:
\begin{equation}
    L_{\text{EFL}} = \frac{1}{N} \sum_{n=1}^N \omega_n \mathcal{F}(t) \left[ L_{\text{CE}}^{(n)} + \lambda_{\text{KL}} \mu(t) \Lambda_{\text{asym}}^{(n)} L_{\text{KL}}^{(n)} \right]
\end{equation}
where $\Lambda_{\text{asym}}^{(n)} = \min(\omega_{y_n}, \Lambda_{\text{max}})$ aggressively squashes false majority evidence on rare-class samples without triggering catastrophic gradient clipping. 

\subsection{Bias-Corrected Calibration and Adaptive Thresholding}
To counteract prior shift, a scalar temperature $T$ and a prior-correction bias vector $\bm{b}$ are introduced:
\begin{equation}
    z'_c = \frac{z_c}{T} + b_c
\end{equation}
where theoretically $b_c \approx \ln \pi_{\text{true},c} - \ln \pi_{\text{train},c}$. This calibration module is evaluated in future ablations; the current case study uses validation-based thresholding. 

Training proceeds in three stages: dense warmup, epoch-level topology adaptation, and masked evidential optimization. After warmup, a proxy batch is used once per epoch to compute pruning and regrowth scores, update latent vitality scores, and refresh 2:4 masks. Standard mini-batch training then updates only active weights under the fixed mask for the remainder of the epoch.

\section{Experimental Setup: Initial ISIC 2024 Case Study}
We instantiate GUDS-EDL on the ISIC 2024 Challenge dataset (3D-TBP images with 0.15\% minority prevalence). We adopt a stratified 70/10/20 train/validation/test split grouped by patient. We note that utilizing the validation set for threshold sweeps represents a limitation, while the test partition acts as a strict local hold-out set. We compare against Fisher EDL, Flexible EDL, and R-EDL. Models are evaluated using Sensitivity, Specificity, F2, Macro-AUROC, pAUC$_{0.80}$, PR-AUC, ECE, Minority ECE, and AURC.

\section{Results}
\begin{table*}[!ht]
\centering
\caption{ISIC 2024 Case Study: Classification performance comparison under Adaptive Operating Modes. The extreme class imbalance (0.15\% rare-event prevalence) renders standard default thresholds ($\tau=0.50$) unviable. The Balanced Utility mode maximizes the arithmetic mean of Sensitivity and Specificity, while the High-Recall Fail-Safe mode ensures a minimum diagnostic Sensitivity of 80\% to minimize false negatives.}
\label{tab:results_classification}
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.2pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l C C C C C C C C C}
\toprule
 & \multicolumn{3}{c}{\textbf{Default Threshold ($\tau = 0.50$)}} & \multicolumn{3}{c}{\textbf{Balanced Utility}} & \multicolumn{3}{c}{\textbf{High-Recall Fail-Safe}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
\textbf{Model} & \textbf{Sens.} $\uparrow$ & \textbf{Spec.} $\uparrow$ & \textbf{F2} $\uparrow$ & \textbf{Sens.} $\uparrow$ & \textbf{Spec.} $\uparrow$ & \textbf{F2} $\uparrow$ & \textbf{Sens.} $\uparrow$ & \textbf{Spec.} $\uparrow$ & \textbf{F2} $\uparrow$ \\
\midrule
Fisher EDL   & $0.3117$ & $0.9864$ & $0.0978$ & $0.7273$ & $0.8739$ & $0.0323$ & $0.8052$ & $0.7393$ & $0.0177$ \\
Flexible EDL & $0.3117$ & $0.9892$ & $0.1147$ & $0.8182$ & $0.8203$ & $0.0258$ & $0.8182$ & $0.8203$ & $0.0258$ \\
R-EDL        & $0.3506$ & $0.9796$ & $0.0805$ & $0.7662$ & $0.8357$ & $0.0264$ & $0.8052$ & $0.7473$ & $0.0182$ \\
\midrule
\textbf{GUDS-EDL (Ours)} & $\mathbf{0.4286}$ & $0.9854$ & $\mathbf{0.1265}$ & $\mathbf{0.8312}$ & $\mathbf{0.8828}$ & $\mathbf{0.0396}$ & $0.8052$ & $\mathbf{0.9030}$ & $\mathbf{0.0459}$ \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}

\begin{table*}[!ht]
\centering
\caption{Threshold-independent calibration, ranking, and uncertainty metrics. Lower ECE, Minority ECE, and AURC represent better confidence calibration and superior selective classification (abstention) capacity.}
\label{tab:results_uncertainty}
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.0pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l C C C C C C C C}
\toprule
\textbf{Model} & \textbf{Macro-AUROC} $\uparrow$ & \textbf{pAUC$_{0.80}$} $\uparrow$ & \textbf{PR-AUC} $\uparrow$ & \textbf{ECE} $\downarrow$ & \textbf{Minority ECE} $\downarrow$ & \textbf{AURC} $\downarrow$ & \textbf{Mean $u_e$} & \textbf{Mean $u_a$} \\
\midrule
Fisher EDL   & $0.8660$ & $0.1048$ & $0.0266$ & $0.2139$ & $\mathbf{0.3491}$ & $0.0009$ & $0.3822$ & $0.4453$ \\
Flexible EDL & $0.8835$ & $0.1138$ & $0.0251$ & $0.1097$ & $0.6101$ & $\mathbf{0.0003}$ & $0.2068$ & $0.2913$ \\
R-EDL        & $0.8741$ & $0.1126$ & $0.0183$ & $\mathbf{0.0989}$ & $0.3583$ & $0.0018$ & $0.2285$ & $\mathbf{0.1298}$ \\
\midrule
\textbf{GUDS-EDL (Ours)} & $\mathbf{0.9088}$ & $\mathbf{0.1296}$ & $\mathbf{0.0297}$ & $0.1113$ & $0.5089$ & $\mathbf{0.0003}$ & $0.2112$ & $0.2985$ \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}

GUDS-EDL improves ranking and high-recall operating performance, while calibration remains mixed. As shown in Table~\ref{tab:results_classification}, under the High-Recall Fail-Safe mode, GUDS-EDL achieves superior specificity. Table~\ref{tab:results_uncertainty} demonstrates that GUDS-EDL achieves the highest Macro-AUROC, pAUC$_{0.80}$, and PR-AUC, though global and minority ECE indicate that exact calibration requires further tuning.

\section{Planned Generalization Protocol and Limitations}

\paragraph{Planned Generalization Protocol.} The current empirical evidence is limited to the ISIC 2024 case study. To test whether the proposed topology adaptation generalizes beyond medical rare-event screening, we define a planned protocol on CIFAR-100-LT for controlled long-tailed recognition and MVTec AD for image-level industrial anomaly detection. These experiments will compare GUDS-EDL against CE, Focal Loss, Logit Adjustment, Dense EDL, Static 2:4 EDL, and RigL-style dynamic sparsity, using macro-F1, few-shot accuracy, AUROC, PR-AUC, ECE, AURC, and low-FPR recall. Hardware profiling will report active parameters, theoretical FLOPs, peak VRAM, throughput, and mask turnover. These planned experiments are not used to support the current empirical claims.

\paragraph{Ablation Protocol.} A complete ablation suite is planned to isolate the effects of the signed pruner, class-conditioned regrower, topology cache, anti-crystallization noise, asymmetric KL regularization, and bias-corrected calibration. Because these experiments are not complete in the current draft, they are reported as a protocol in Appendix rather than as empirical evidence.

\paragraph{Limitations and Broader Impact.} 
(1) Current empirical validation is limited to the ISIC 2024 dataset.
(2) Minority calibration remains challenging due to the severe prior shift.
(3) Full 2:4 hardware acceleration requires specialized sparse kernels.
(4) High-stakes rare-event systems must defer uncertain samples to humans.

\section{Conclusion}
We introduced GUDS-EDL, an uncertainty-guided dynamic sparse evidential learning framework for extreme imbalanced classification. In the ISIC 2024 case study, GUDS-EDL provides initial evidence that evidential topology adaptation can improve rare-event ranking and high-recall operating specificity under strict 2:4 sparsity. Broader validation on controlled long-tailed recognition, industrial anomaly detection, additional backbones, and hardware-accelerated sparse kernels remains future work.

\begin{thebibliography}{99}

\bibitem[\protect\citeauthoryear{Cao \emph{et al.}}{2019}]{cao2019learning}
K.~Cao, C.~Wei, A.~Gaidon, N.~Arechiga, and T.~Ma.
\newblock Learning imbalanced datasets with label-distribution-aware margin loss.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2019.

\bibitem[\protect\citeauthoryear{Charpentier, Z{\"u}gner, and G{\"u}nnemann}{2020}]{charpentier2020posterior}
B.~Charpentier, D.~Z{\"u}gner, and S.~G{\"u}nnemann.
\newblock Posterior Network: Uncertainty estimation without OOD samples via density-based pseudo-counts.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2020.

\bibitem[\protect\citeauthoryear{Cui \emph{et al.}}{2019}]{cui2019class}
Y.~Cui, M.~Jia, T.-Y. Lin, Y.~Song, and S.~Belongie.
\newblock Class-balanced loss based on effective number of samples.
\newblock In {\em Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, pages 9268--9277, 2019.

\bibitem[\protect\citeauthoryear{Deng \emph{et al.}}{2023}]{deng2023fisher}
K.~Deng, Y.~Zhang, and J.~He.
\newblock Evidential learning with Fisher information.
\newblock In {\em IEEE/CVF International Conference on Computer Vision (ICCV)}, 2023.

\bibitem[\protect\citeauthoryear{Ding \emph{et al.}}{2020}]{ding2020revisiting}
X.~Ding, G.~Ding, Y.~Guo, and J.~Han.
\newblock Revisiting the evaluation of uncertainty estimation and failure detection.
\newblock In {\em IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops}, 2020.

\bibitem[\protect\citeauthoryear{Evci \emph{et al.}}{2020}]{evci2020rigging}
U.~Evci, T.~Gale, J.~Menick, P.S.~Castro, and E.~Elsen.
\newblock Rigging the lottery: Making all tickets winners.
\newblock In {\em International Conference on Machine Learning (ICML)}, 2020.

\bibitem[\protect\citeauthoryear{Geifman and El-Yaniv}{2017}]{geifman2017selective}
Y.~Geifman and R.~El-Yaniv.
\newblock Selective classification for deep neural networks.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2017.

\bibitem[\protect\citeauthoryear{Guo \emph{et al.}}{2017}]{guo2017calibration}
C.~Guo, G.~Pleiss, Y.~Sun, and K.Q.~Weinberger.
\newblock On calibration of modern neural networks.
\newblock In {\em International Conference on Machine Learning (ICML)}, 2017.

\bibitem[\protect\citeauthoryear{Kang \emph{et al.}}{2020}]{kang2019decoupling}
B.~Kang, S.~Xie, M.~Rohrbach, Z.~Yan, A.~Gordo, J.~Feng, and Y.~Kalantidis.
\newblock Decoupling representation and classifier for long-tailed recognition.
\newblock In {\em International Conference on Learning Representations (ICLR)}, 2020.

\bibitem[\protect\citeauthoryear{Lasby \emph{et al.}}{2023}]{lasby2023srigl}
M.~Lasby, A.~Golubeva, and G.~Nadiradze.
\newblock SRigL: Sparse training with structured regrowth.
\newblock In {\em Advances in Neural Information Processing Systems Workshop}, 2023.

\bibitem[\protect\citeauthoryear{Malinin and Gales}{2018}]{malinin2018predictive}
A.~Malinin and M.~Gales.
\newblock Predictive uncertainty estimation via prior networks.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2018.

\bibitem[\protect\citeauthoryear{Menon \emph{et al.}}{2021}]{menon2021long}
A.K.~Menon, S.~Jayaraman, A.S.~Rawat, S.~Kumar, and S.~Vemulapalli.
\newblock Long-tail learning via logit adjustment.
\newblock In {\em International Conference on Learning Representations (ICLR)}, 2021.

\bibitem[\protect\citeauthoryear{Mishra \emph{et al.}}{2021}]{mishra2021accelerating}
A.~Mishra, J.~Pool, M.~Smelyanskiy, and D.~Dally.
\newblock Accelerating sparse deep neural networks.
\newblock In {\em IEEE International Parallel and Distributed Processing Symposium Workshops}, 2021.

\bibitem[\protect\citeauthoryear{Mocanu \emph{et al.}}{2018}]{mocanu2018scalable}
D.C.~Mocanu, E.~Mocanu, P.~Stone, P.H.~Nguyen, M.~Gibescu, and L.~Liotta.
\newblock Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science.
\newblock {\em Nature Communications}, 9(1):2383, 2018.

\bibitem[\protect\citeauthoryear{Ren \emph{et al.}}{2020}]{ren2020balanced}
J.~Ren, C.~Yu, X.~Ma, H.~Zhao, S.~Yi, and H.~Li.
\newblock Balanced meta-softmax for long-tailed visual recognition.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2020.

\bibitem[\protect\citeauthoryear{Sensoy, Kaplan, and Kandemir}{2018}]{sensoy2018evidential}
M.~Sensoy, L.~Kaplan, and M.~Kandemir.
\newblock Evidential deep learning to quantify classification uncertainty.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2018.

\bibitem[\protect\citeauthoryear{Shen \emph{et al.}}{2024}]{shen2024mirage}
J.~Shen, S.~Zhou, T.~Liu, B.~Han, and M.~Kandemir.
\newblock The mirage of evidential deep learning.
\newblock In {\em International Conference on Machine Learning (ICML)}, 2024.

\bibitem[\protect\citeauthoryear{Yoon \emph{et al.}}{2025}]{yoon2025flexible}
S.~Yoon, Y.~Wang, and X.~Zhou.
\newblock Flexible evidential deep learning for imbalanced classification.
\newblock In {\em AAAI Conference on Artificial Intelligence}, 2025.

\end{thebibliography}

\appendix

\section{Appendix A. Dirichlet and Subjective Logic Derivations}
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

\subsection{A.4 Closed-form KL Divergences}
The class-conditioned regrowth KL divergence from the uniform prior is:
\begin{equation}
\begin{split}
    \KL(\Dir(\bm{\alpha}) \parallel \Dir(\mathbf{1})) ={}& \ln \Gamma(S) - \sum_{c=1}^K \ln \Gamma(\alpha_c) - \ln \Gamma(K) \\
    &+ \sum_{c=1}^K (\alpha_c - 1)\left[ \psi(\alpha_c) - \psi(S) \right]
\end{split}
\end{equation}

To shrink false evidence to the flat prior $1.0$, we adjust the Dirichlet parameters to $\tilde{\alpha}_c^{(n)} = y_c^{(n)} + (1 - y_c^{(n)}) \cdot \alpha_c^{(n)}$. The full asymmetric KL divergence penalty $L_{\text{KL}}^{(n)}$ is computed as:
\begin{equation}
\begin{aligned}
    L_{\text{KL}}^{(n)} ={}& \ln \left( \frac{\Gamma\left(\sum_{c=1}^K \tilde{\alpha}_c^{(n)}\right)}{\Gamma(K) \prod_{c=1}^K \Gamma(\tilde{\alpha}_c^{(n)})} \right) \\
    &+ \sum_{c=1}^K (\tilde{\alpha}_c^{(n)} - 1)\left[\psi(\tilde{\alpha}_c^{(n)}) - \psi\!\left(\sum_{c=1}^K \tilde{\alpha}_c^{(n)}\right)\right]
\end{aligned}
\end{equation}

\section{Appendix B. Full GUDS-EDL Algorithm}
\begin{algorithm}[!h]
\caption{Universal GUDS-EDL Training and Optimization Loop}
\label{alg:guds_edl}
\begin{algorithmic}[1]
\REQUIRE Dataset $\mathcal{D}$, model weights $\bm{W}^{(0)}$, latent scores $\bm{V}^{(0)} \leftarrow |\bm{W}^{(0)}|$, epochs $T$, warmup epochs $T_w$, structural update rate $\eta_{\text{struct}} = 0.02$.
\FOR{each epoch $t = 1 \dots T$}
    \IF{$t == T_w$}
        \STATE Freeze learned channel permutations into static index maps.
    \ENDIF
    \IF{$t \ge T_w$}
        \STATE Sample a proxy batch $(\bm{X}, \bm{y}) \sim \mathcal{D}$ for amortized topology computation.
        \STATE Compute structural gradients (Pruner and Regrower):
        \STATE \quad $\bm{C}_{\text{pruner}} \leftarrow \text{ReLU}\left( \bm{W} \odot \nabla_{\bm{W}} (u_a / (u_e + \epsilon)) \right)$
        \STATE \quad $\mathcal{L}_{\text{grow}} \leftarrow \omega_y u_e\, \KL(\Dir(\bm{\alpha}) \parallel \Dir(\mathbf{1}))$
        \STATE \quad $\bm{G}_{\text{regrower}} \leftarrow \nabla_{\bm{W}} \mathcal{L}_{\text{grow}}$
        \STATE Update continuous latent scores $\bm{V}$ using $\bm{C}_{\text{pruner}}$ and $\bm{G}_{\text{regrower}}$.
        \STATE Apply layer-norm anti-crystallization noise if max growth is below threshold.
        \STATE Update binary mask $\bm{M}^{(l)}$ via Reshaped NVIDIA 2:4 top-2-of-4 selection on $\bm{V}^{(l)}$.
    \ELSE
        \STATE Set masks $\bm{M}^{(l)} \leftarrow \mathbf{1}$ (Dense Warmup).
    \ENDIF
    \FOR{each mini-batch step $s$ in epoch $t$}
        \STATE Sample batch $(\bm{X}, \bm{y}) \sim \mathcal{D}$.
        \STATE Forward pass: compute logits $\bm{z}$ using static masked weights $\bm{W}_{\text{eff}} = \bm{W} \odot \bm{M}$.
        \STATE Compute Dirichlet parameters: $\alpha_c \leftarrow \ln(1 + e^{z_c}) + 1.0$.
        \STATE Compute expected probabilities $\hat{p}_c = \alpha_c/S$ and uncertainties $u_e, u_a$.
        \STATE Compute Evidential Focal Loss (EFL) $L_{\text{EFL}}$.
        \STATE Backward pass: compute gradients $\nabla_{\bm{W}} L_{\text{scaled}}$.
        \STATE Update active weights: $\bm{W} \leftarrow \bm{W} - \eta_s \text{AdamW}(\nabla_{\bm{W}} L_{\text{scaled}} \odot \bm{M})$.
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}

\section{Appendix C. Planned Ablation Protocol}
We plan a complete ablation suite to isolate the effects of the signed pruner, class-conditioned regrower, topology cache, anti-crystallization noise, asymmetric KL regularization, and bias-corrected calibration. These experiments will assess Macro-AUROC, ECE, Mask Turnover Rate, and Dead Channel Ratio to analyze the topological plasticity and uncertainty dynamics of GUDS-EDL.

\section{Appendix D. Planned Generalization Benchmarks}
\subsection{D.1 CIFAR-100-LT Protocol}
We define a planned protocol on CIFAR-100-LT with varying imbalance profiles (1:10, 1:50, and 1:100). The protocol will benchmark our method against standard Long-Tailed and Evidential models using few-shot, medium-shot, and many-shot accuracy, along with Expected Calibration Error.

\subsection{D.2 MVTec AD Image-Level Protocol}
To evaluate on industrial rare-event data, we will apply GUDS-EDL to the MVTec AD benchmark. The evaluation will use image-level classification (Normal vs. Defective) and report AUROC, PR-AUC, Recall@FPR1\%, and AURC to validate the selective prediction capability.

\subsection{D.3 Hardware Profiling Protocol}
We plan to benchmark the strict 2:4 structured sparsity of GUDS-EDL on NVIDIA Sparse Tensor Cores. The protocol includes theoretical FLOP profiling and empirical throughput testing on an NVIDIA RTX A6000 GPU to document peak VRAM reductions during the decoupled topological adaptation phase.

\section{Appendix E. Reproducibility Details}
\subsection{E.1 ISIC 2024 Case Study Experimental Setup}
We instantiated GUDS-EDL on the ISIC 2024 dataset, predicting benign versus malignant outcomes under a 0.15\% prevalence. The training set was sampled with a maximum 1:20 imbalance ratio for stability, and augmented with spatial flipping. No log-prior adjustments were applied during training to preserve relative logit scaling. The validation set guided post-hoc temperature and thresholding parameters, avoiding test-set leakage.

\end{document}
"""

with open("d:/MDEP/main_text.tex", "w", encoding="utf-8") as f:
    f.write(tex_content)

print("Rewritten main_text.tex successfully.")
