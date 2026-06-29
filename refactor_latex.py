import re

def extract_block(text, tag):
    pattern = r'\\begin\{' + tag + r'\}(.*?)\\end\{' + tag + r'\}'
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(0) for match in matches]

def refactor():
    with open('main_text.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all tables
    tables = extract_block(content, 'table\*?')
    
    # Extract figure
    figures = extract_block(content, 'figure\*?')
    fig1 = figures[0] if figures else ""
    
    # Extract Algorithm
    algorithms = extract_block(content, 'algorithm')
    alg1 = algorithms[0] if algorithms else ""
    
    # Extract Bibliography
    bib = extract_block(content, 'thebibliography')
    bib_block = bib[0] if bib else ""
    
    # Identify tables
    tab_modes = ""
    tab_metrics = ""
    tab_config = ""
    tab_qg = ""
    tab_ablation = ""
    tab_cifar = ""
    tab_mvtec = ""
    tab_hw = ""
    
    for t in tables:
        if 'Adaptive Operating Modes' in t or 'tab:results_classification' in t:
            tab_modes = t
        elif 'Threshold-independent' in t or 'tab:results_uncertainty' in t:
            tab_metrics = t
        elif 'System Optimization Configuration' in t or 'tab:config' in t:
            tab_config = t
        elif 'Quality-Gated' in t or 'tab:quality_gated' in t:
            tab_qg = t
        elif 'Ablation' in t or 'tab:ablation' in t:
            tab_ablation = t
        elif 'CIFAR-100-LT' in t or 'tab:cifar' in t:
            tab_cifar = t
        elif 'MVTec AD' in t or 'tab:mvtec' in t:
            tab_mvtec = t
        elif 'Hardware' in t or 'tab:hardware' in t:
            tab_hw = t

    preamble_match = re.search(r'(.*?\\begin\{document\})', content, re.DOTALL)
    preamble = preamble_match.group(1) if preamble_match else ""

    # Generate the new content
    new_tex = preamble + "\n\\maketitle\n\n"
    
    # Abstract
    new_tex += r"""\begin{abstract}
Extreme imbalanced classification arises in industrial defect inspection, fraud detection, autonomous driving edge-case recognition, and safety-critical medical screening. Existing long-tailed methods primarily rebalance losses or classifiers, evidential models estimate uncertainty without changing network structure, and dynamic sparse training methods evolve connectivity using magnitude or task-gradient heuristics that are blind to uncertainty. We introduce GUDS-EDL, a generalized uncertainty-guided dynamic sparse evidential learning framework for extreme imbalanced classification. GUDS-EDL couples a Dirichlet evidential prediction head with strict NVIDIA 2:4 structured sparsity, allowing uncertainty to shape not only the model's output confidence but also its internal topology. A signed first-order uncertainty-guided pruner removes connections whose removal is estimated to reduce evidential risk, while a class-conditioned evidence-seeking regrower restores capacity in representation-deficient regions with high vacuity and rare-class utility. To stabilize learning under severe prior shift, we further introduce bounded asymmetric evidential regularization and bias-corrected post-hoc calibration. We instantiate GUDS-EDL on ISIC 2024 as an initial high-stakes rare-event case study, where it improves rare-event ranking and high-recall operating specificity over representative evidential baselines. Beyond this case study, we define a planned evaluation protocol for controlled long-tailed recognition and industrial image-level anomaly detection.
\end{abstract}

\section{Introduction}
Real-world machine learning systems rarely operate under balanced, clean, and equally costly class distributions. In many high-impact domains, the events that matter most are precisely those that appear least often. This extreme imbalance creates a dual challenge: standard dense networks overfit majority-class patterns, and they produce overconfident predictions on rare or ambiguous samples, making them difficult to trust in mission-critical settings.

Long-tailed learning methods address part of this problem through re-sampling or re-weighting, but do not adapt the network's internal topology. Evidential Deep Learning replaces softmax point confidence with a Dirichlet distribution over class probabilities, providing uncertainty signals in a single forward pass, but leaves the underlying architecture unchanged. Dynamic Sparse Training offers a complementary route by learning sparse subnetworks from scratch, but existing methods use magnitude or task-gradient heuristics that are blind to uncertainty. 

To address this gap, we propose \textbf{GUDS-EDL: Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning}. GUDS-EDL combines a Dirichlet evidential head with strict NVIDIA 2:4 structured sparsity. Its topology is governed by a signed first-order uncertainty-guided pruner that removes noise-amplifying connections using evidential risk, and a class-conditioned evidence-seeking regrower that restores capacity in high-vacuity, representation-deficient regions. To stabilize this process under severe imbalance, we introduce bounded asymmetric evidential regularization and bias-corrected temperature scaling.

Our contributions are: (1) We introduce an uncertainty-guided dynamic sparse training mechanism that uses evidential risk and vacuity to prune and regrow 2:4 structured connections. (2) We propose an imbalance-aware evidential optimization objective with bounded asymmetric regularization and bias-corrected calibration. (3) We provide an initial high-stakes ISIC 2024 rare-event case study and outline a planned multi-benchmark protocol for broader validation.

\section{Related Work}
\paragraph{Long-Tailed and Imbalanced Learning.} Long-tailed recognition has been extensively studied through objective re-weighting and classifier balancing~\cite{cui2019class,cao2019learning,menon2021long,ren2020balanced}. Decoupling representation learning from classifier training further improves boundaries~\cite{kang2019decoupling}. However, these methods primarily modify the loss landscape rather than dynamically adapting the network's internal sparse capacity toward minority concepts.

\paragraph{Evidential and Dirichlet Uncertainty.} Evidential Deep Learning~\cite{sensoy2018evidential} and related Prior/Posterior Networks~\cite{malinin2018predictive} formulate classification via a Dirichlet distribution, enabling single-pass uncertainty estimation. Recent variants like Fisher EDL~\cite{deng2023fisher} and Flexible EDL~\cite{flexible_edl2023,yoon2025flexible} improve calibration, while recent critiques caution against over-interpreting EDL as true Bayesian uncertainty~\cite{shen2024mirage}.

\paragraph{Calibration and Selective Prediction.} Reliable deployment requires calibrated confidence~\cite{guo2017calibration} and the ability to abstain on ambiguous samples, formalized by selective classification and Area Under the Risk-Coverage Curve (AURC)~\cite{geifman2017selective,geifman2019selectivenet,ding2020revisiting}.

\paragraph{Dynamic Sparse Training and Structured Sparsity.} Dynamic sparse training maintains sparse connectivity throughout optimization~\cite{dettmers2019sparse,evci2020rigging}. Modern hardware considerations motivate strict structured sparsity patterns, such as NVIDIA's 2:4 constraint~\cite{mishra2021accelerating,lasby2023srigl}.

Existing long-tailed methods rebalance objectives or classifiers, existing evidential methods estimate uncertainty without changing topology, and existing DST methods evolve masks without uncertainty awareness. GUDS-EDL fills this gap by using evidential uncertainty itself as the signal for structured sparse topology adaptation under extreme imbalance.

\section{Method}

\subsection{Evidential Prediction and Uncertainty Proxies}
We parameterize the non-negative evidence vector $\bm{e}$ from logits $\bm{z}$ using the Softplus activation: $e_c = \ln(1 + \exp(z_c))$. The Dirichlet concentration parameters $\bm{\alpha}$ are defined as $\alpha_c = e_c + 1.0$. The Dirichlet strength is $S = \sum_c \alpha_c$, and the expected probability is $\hat{p}_c = \alpha_c / S$. We define two uncertainty proxies to guide structural adaptation: Dirichlet Vacuity $u_e = K / S$, and Expected Categorical Entropy $u_a = \sum_c \frac{\alpha_c}{S}[\psi(S+1) - \psi(\alpha_c+1)]$. Formal derivations and uncertainty bounds are provided in the Appendix.

"""
    new_tex += fig1 + "\n\n"
    
    new_tex += r"""
\subsection{2:4 Dynamic Sparse Topology}
We enforce NVIDIA 2:4 structured sparsity by defining two tiers of variables: active weights $\bm{W}$ optimized via task loss, and continuous latent vitality scores $\bm{V}$ used to select the top-2 scores per 4-weight block. Masks are updated once per epoch via an amortized proxy batch, decoupling topology evolution from high-frequency mini-batch noise.

\subsection{Signed Uncertainty-Guided Pruning}
We define the evidential risk ratio $R = \frac{u_a}{u_e + \epsilon}$. The pruning criterion targets connections whose removal strictly decreases this risk ratio, evaluating the Signed First-Order Removal Effect: $C_{ij} = \text{Rank}(\text{ReLU}(w_{ij} \frac{\partial R}{\partial w_{ij}}))$. This criterion is designed to avoid pruning connections whose removal would increase evidential risk.

\subsection{Class-Conditioned Evidence-Seeking Regrowth}
To enforce targeted regrowth, we formulate an uncertainty-gated growth objective that gates the KL divergence from a Uniform Prior by the ground-truth class weight $\omega_y$ and vacuity $u_e$: $L_{\text{grow}} = \omega_y \cdot u_e \cdot \text{KL}(\text{Dir}(\bm{\alpha}) \parallel \text{Dir}(\mathbf{1}))$. The regrowth signal is $G_{ij} = \text{Rank}(|\frac{\partial L_{\text{grow}}}{\partial w_{ij}}|)$. The full KL closed-form expression is provided in the Appendix.

\subsection{Imbalance-Aware Evidential Objective}
We integrate an Evidential Focal Loss with square-root frequency dampening and a Bounded Asymmetric KL Penalty:
\begin{equation}
    L_{\text{EFL}} = \frac{1}{N} \sum_{n=1}^N \omega_n \mathcal{F}(t) \left[ L_{\text{CE}}^{(n)} + \lambda_{\text{KL}} \mu(t) \Lambda_{\text{asym}}^{(n)} L_{\text{KL}}^{(n)} \right]
\end{equation}
where $\Lambda_{\text{asym}}^{(n)} = \min(\omega_{y_n}, \Lambda_{\text{max}})$ squashes false majority evidence on rare-class samples without triggering catastrophic gradient clipping. 

\subsection{Bias-Corrected Calibration and Adaptive Thresholding}
To counteract prior shift, a scalar temperature $T$ and a prior-correction bias vector $\bm{b}$ are optimized post-hoc: $z'_c = z_c/T + b_c$, where initialized $b_c \approx \ln \pi_{\text{true},c} - \ln \pi_{\text{train},c}$. This calibration module is evaluated in future ablations; the current case study uses validation-based thresholding. Training proceeds via dense warmup, epoch-level topology adaptation, and masked evidential optimization, detailed in the Appendix.

\section{Experimental Setup: Initial ISIC 2024 Case Study}
We instantiate our method on the ISIC 2024 Challenge dataset (3D-TBP images with 0.15\% minority prevalence). We adopt a stratified 70/10/20 train/validation/test split grouped by patient. We note that utilizing the validation set for threshold sweeps represents a limitation, while the test partition acts as a local hold-out set. We compare against Fisher EDL, Flexible EDL, and R-EDL. Models are evaluated using Sensitivity, Specificity, F2, Macro-AUROC, pAUC$_{0.80}$, PR-AUC, ECE, Minority ECE, and AURC.

\section{Results}
GUDS-EDL improves ranking and high-recall operating performance, while calibration remains mixed.
"""
    new_tex += tab_modes + "\n" + tab_metrics + "\n"
    
    new_tex += r"""
\section{Planned Generalization Protocol and Limitations}
\paragraph{Planned Generalization Protocol.} The current empirical evidence is limited to the ISIC 2024 case study. To test whether the proposed topology adaptation generalizes beyond medical rare-event screening, we define a planned protocol on CIFAR-100-LT for controlled long-tailed recognition and MVTec AD for image-level industrial anomaly detection. These experiments will compare GUDS-EDL against CE, Focal Loss, Logit Adjustment, Dense EDL, Static 2:4 EDL, and RigL-style dynamic sparsity. Hardware profiling will report active parameters, theoretical FLOPs, peak VRAM, and throughput. A complete ablation suite is also planned to isolate component effects. Because these experiments are not complete in the current draft, they are reported as protocols in the Appendix rather than as empirical evidence.

\paragraph{Limitations and Broader Impact.} (1) Current empirical validation is limited to ISIC 2024. (2) Minority calibration remains challenging despite macro-level improvements. (3) Full 2:4 hardware acceleration requires specialized sparse kernels for inference. (4) High-stakes rare-event systems must defer uncertain samples to humans.

\section{Conclusion}
We introduced GUDS-EDL, an uncertainty-guided dynamic sparse evidential learning framework for extreme imbalanced classification. In the ISIC 2024 case study, GUDS-EDL provides initial evidence that evidential topology adaptation can improve rare-event ranking and high-recall operating specificity under strict 2:4 sparsity. Broader validation on controlled long-tailed recognition, industrial anomaly detection, additional backbones, and hardware-accelerated sparse kernels remains future work.

"""
    new_tex += bib_block + "\n\n"
    new_tex += r"""\appendix

\section{Appendix A: Dirichlet and Subjective Logic Derivations}
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

\section{Appendix B: Full GUDS-EDL Algorithm}
"""
    new_tex += alg1 + "\n\n"
    
    new_tex += r"""
\section{Appendix C: Planned Ablation Protocol}
We plan a complete ablation suite to isolate the effects of the signed pruner, class-conditioned regrower, topology cache, anti-crystallization noise, asymmetric KL regularization, and bias-corrected calibration.
"""
    # Replace caption of ablation table
    if tab_ablation:
        tab_ablation = re.sub(r'\\caption\{.*?\}', r'\\caption{Planned ablation suite. Values will be filled after experiments are complete.}', tab_ablation, flags=re.DOTALL)
        new_tex += tab_ablation + "\n"

    new_tex += r"""
\section{Appendix D: Planned Generalization Benchmarks}
\subsection{D.1 CIFAR-100-LT Protocol}
"""
    if tab_cifar:
        tab_cifar = re.sub(r'\\caption\{.*?\}', r'\\caption{Planned protocol for CIFAR-100-LT. Values will be filled after experiments.}', tab_cifar, flags=re.DOTALL)
        new_tex += tab_cifar + "\n"

    new_tex += r"""\subsection{D.2 MVTec AD Image-Level Protocol}"""
    if tab_mvtec:
        tab_mvtec = re.sub(r'\\caption\{.*?\}', r'\\caption{Planned protocol for MVTec AD. Values will be filled after experiments.}', tab_mvtec, flags=re.DOTALL)
        new_tex += tab_mvtec + "\n"
        
    new_tex += r"""\subsection{D.3 Hardware Profiling Protocol}"""
    if tab_hw:
        tab_hw = re.sub(r'\\caption\{.*?\}', r'\\caption{Planned hardware profiling. Values will be filled after tests.}', tab_hw, flags=re.DOTALL)
        new_tex += tab_hw + "\n"
        
    new_tex += r"""\subsection{D.4 Quality-Gated Failure Detection Protocol}"""
    if tab_qg:
        tab_qg = re.sub(r'\\caption\{.*?\}', r'\\caption{Planned quality-gated failure detection suite. Values will be filled after experiments.}', tab_qg, flags=re.DOTALL)
        new_tex += tab_qg + "\n"

    new_tex += r"""
\section{Appendix E: Reproducibility Details}
"""
    new_tex += tab_config + "\n\n"
    new_tex += r"\end{document}"
    
    with open('main_text_refactored.tex', 'w', encoding='utf-8') as f:
        f.write(new_tex)
        
refactor()
