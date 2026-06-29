with open(r'd:\MDEP\main_text.tex', 'r', encoding='utf-8') as f:
    content = f.read()

def extract_between(text, start_str, end_str):
    start_idx = text.find(start_str)
    if start_idx == -1: return ""
    end_idx = text.find(end_str, start_idx + len(start_str))
    if end_idx == -1: return ""
    return text[start_idx:end_idx + len(end_str)]

# Preamble and Title
idx_maketitle = content.find('\\maketitle') + len('\\maketitle')
preamble = content[:idx_maketitle]

# Related Work
idx_related_start = content.find('\\section{Related Work}')
idx_math_start = content.find('\\section{Mathematical Foundations')
related_work = content[idx_related_start:idx_math_start]

# Table 1
table1 = extract_between(content, '\\begin{table}[t]\n\\centering\n\\caption{Adaptive Operating Modes', '\\end{table}')
# Table 2
table2 = extract_between(content, '\\begin{table}[t]\n\\centering\n\\caption{Threshold-independent Predictive Performance', '\\end{table}')

# Appendix tables
table3 = extract_between(content, '\\begin{table}[t]\n\\centering\n\\caption{Quality-Gated Selective Prediction', '\\end{table}')
table4 = extract_between(content, '\\begin{table*}[t]\n\\centering\n\\caption{Ablation Study', '\\end{table*}')
table5 = extract_between(content, '\\begin{table}[t]\n\\centering\n\\caption{CIFAR-100-LT', '\\end{table}')
table6 = extract_between(content, '\\begin{table}[t]\n\\centering\n\\caption{MVTec AD', '\\end{table}')
table7 = extract_between(content, '\\begin{table}[t]\n\\centering\n\\caption{Hardware Efficiency', '\\end{table}')

# Algorithm
algorithm1 = extract_between(content, '\\begin{algorithm}[t]', '\\end{algorithm}')

# Bibliography
idx_bib_start = content.find('\\begin{thebibliography}')
idx_bib_end = content.find('\\end{thebibliography}') + len('\\end{thebibliography}')
bibliography = content[idx_bib_start:idx_bib_end]

new_content = f"""{preamble}

\\begin{{abstract}}
Extreme imbalanced classification arises in real-world rare-event domains, including industrial defect detection, fraud detection, autonomous driving edge cases, and safety-critical medical screening. Standard dense neural networks often overfit majority classes and produce overconfident predictions on rare classes, while conventional sparse training methods ignore uncertainty signals.

We propose \\textbf{{GUDS-EDL}}, an uncertainty-guided dynamic sparse evidential learning framework for extreme class imbalance. GUDS-EDL combines Dirichlet evidential learning with strict NVIDIA 2:4 structured sparsity. A signed uncertainty-guided pruner removes connections whose removal is estimated to reduce evidential risk, while a class-conditioned evidence-seeking regrower restores capacity in representation-deficient regions. We further introduce bounded asymmetric evidential regularization and bias-corrected temperature scaling to improve the sensitivity-calibration trade-off under severe prior shift.

We instantiate GUDS-EDL on ISIC 2024 as an initial high-stakes rare-event case study. The current draft reports this case study and provides a planned protocol for broader validation on controlled long-tailed recognition and industrial image-level anomaly detection.
\\end{{abstract}}

\\section{{Introduction}}\\label{{sec:intro}}
The deployment of large-scale deep learning models in real-world systems often faces extreme class imbalance, where rare but high-cost events are vastly underrepresented. This long-tailed distribution occurs naturally across critical domains including industrial defect detection, autonomous driving edge cases, and medical abnormalities. In these high-stakes regimes, models must be simultaneously accurate on rare classes, precisely calibrated, and efficient enough for low-latency deployment.

Standard softmax-based neural networks are notoriously vulnerable to these challenges, tending to overfit majority classes and produce overconfident predictions~\\cite{{guo2017calibration}}. While Evidential Deep Learning (EDL)~\\cite{{sensoy2018evidential}} mitigates overconfidence by parameterizing a Dirichlet distribution to represent ignorance (epistemic uncertainty) and data noise (aleatoric uncertainty), existing Dynamic Sparse Training (DST) methods~\\cite{{evci2020rigging}} prune connections based solely on weight magnitudes or task gradients, remaining entirely blind to these uncertainty signals.

To bridge this gap, we introduce \\textbf{{Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning (GUDS-EDL)}}, a framework that integrates Evidential Learning with strict NVIDIA 2:4 structured sparsity. The network topology is dynamically optimized by two computational mechanisms: an \\textbf{{Uncertainty-Guided Pruner}} that removes noise-amplifying connections using aleatoric uncertainty gradients, and an \\textbf{{Evidence-Seeking Regrower}} that restores links at nodes plagued by high epistemic uncertainty. 

Our contributions are:
\\begin{{itemize}}
    \\item We introduce an uncertainty-guided dynamic sparse training mechanism that uses evidential risk and vacuity to prune and regrow 2:4 structured connections.
    \\item We propose an imbalance-aware evidential optimization objective with bounded asymmetric regularization and bias-corrected calibration.
    \\item We provide an initial high-stakes ISIC 2024 rare-event case study and outline a planned multi-benchmark protocol for broader validation.
\\end{{itemize}}

{related_work}

\\section{{Mathematical Foundations}}\\label{{sec:math}}
We replace the traditional softmax activation with mathematical foundations from Subjective Logic~\\cite{{josang2016subjective}}, mapping network outputs to the parameters of a Dirichlet distribution conjugate prior~\\cite{{sensoy2018evidential}}.

For a $K$-class problem, we parameterize the non-negative evidence vector $\\bm{{e}} \\ge \\bm{{0}}$ from logits $\\bm{{z}} \\in \\R^K$ using the Softplus activation: $e_c = \\ln(1 + \\exp(z_c))$. The Dirichlet concentration parameters $\\bm{{\\alpha}}$ are defined by adding a uniform prior to the evidence: $\\alpha_c = e_c + 1.0$. The Dirichlet strength (total evidence) is $S = \\sum_{{c=1}}^K \\alpha_c$, and the expected probability for class $c$ is $\\hat{{p}}_c = \\frac{{\\alpha_c}}{{S}}$.

\\paragraph{{Uncertainty Proxies.}} Following recent critiques of Evidential Deep Learning~\\cite{{shen2024mirage}}, we explicitly state that we do not claim true Bayesian epistemic uncertainty. Instead, we define mathematically rigorous proxies to guide our structural adaptation:
\\begin{{itemize}}
    \\item \\textbf{{Dirichlet Vacuity ($u_e$):}} Models the network's lack of knowledge (e.g., on OOD or rare data), defined as $u_e = \\frac{{K}}{{S}}$.
    \\item \\textbf{{Expected Categorical Entropy ($u_a$):}} Used as a data-ambiguity/noise proxy (aleatoric uncertainty), defined as $u_a = \\sum_{{c=1}}^K \\frac{{\\alpha_c}}{{S}} \\left[\\psi(S + 1) - \\psi(\\alpha_c + 1)\\right]$.
\\end{{itemize}}

Formal derivations, subjective logic identities, and uncertainty bounds are provided in the Appendix.

\\section{{GUDS-EDL Framework}}\\label{{sec:methodology}}
\\subsection{{Dynamic Sparsity under NVIDIA 2:4 Constraint}}
We decouple representational weight learning from network morphology updates by defining two tiers of variables: active weights $\\bm{{W}}^{{(l)}}$ and continuous latent vitality scores $\\bm{{V}}^{{(l)}}$. We enforce the NVIDIA 2:4 structured sparsity constraint, computing a binary mask $\\bm{{M}}^{{(l)}}$ by selecting the top-2 vitality scores in every contiguous block of 4 weights.

\\subsection{{Uncertainty-Guided Topology Adaptation}}
\\paragraph{{Uncertainty-Guided Pruning.}} We define the evidential risk ratio $R = \\frac{{u_a}}{{u_e + \\epsilon}}$. To safely prune connections, we target those whose removal strictly decreases the risk ratio. The pruning force is defined as the Signed First-Order Removal Effect:
\\begin{{equation}}
    C_{{ij}} = \\text{{Rank}}\\left( \\text{{ReLU}}\\left( w_{{ij}} \\cdot \\frac{{\\partial R}}{{\\partial w_{{ij}}}} \\right) \\right)
\\end{{equation}}
This criterion is designed to avoid pruning connections whose removal would increase evidential risk, targeting noise-amplifying connections instead of blindly pruning high-magnitude weights.

\\paragraph{{Evidence-Seeking Regrowth.}} To enforce targeted regrowth on representation-deficient regions, we formulate a Class-Conditioned Uncertainty-Gated growth objective. We gate the KL divergence from a Uniform Prior by the sample's ground-truth class weight $\\omega_y$ and its current Dirichlet vacuity $u_e$:
\\begin{{equation}}
    L_{{\\text{{grow}}}} = \\omega_y \\cdot u_e \\cdot \\KL(\\Dir(\\bm{{\\alpha}}) \\parallel \\Dir(\\mathbf{{1}}))
\\end{{equation}}
The growth gradient for structural recovery is $G_{{ij}} = \\text{{Rank}}\\left( \\left| \\frac{{\\partial L_{{\\text{{grow}}}}}}{{\\partial w_{{ij}}}} \\right| \\right)$. $\\KL(\\Dir(\\bm{{\\alpha}}) \\parallel \\Dir(\\mathbf{{1}}))$ has a closed-form expression provided in the Appendix.

\\paragraph{{Optimization Loop.}} Training proceeds in three stages: dense warmup, epoch-level topology adaptation, and masked evidential optimization. After warmup, a proxy batch is used once per epoch to compute pruning and regrowth scores ($C_{{ij}}, G_{{ij}}$), update latent vitality scores $\\bm{{V}}^{{(l)}}$, and refresh 2:4 masks $\\bm{{M}}^{{(l)}}$. Standard mini-batch training then updates only active weights under the fixed mask for the remainder of the epoch. The full Algorithm is provided in the Appendix.

\\section{{Optimization Framework}}\\label{{sec:optimization}}
We integrate an Evidential Focal Loss (EFL) with square-root frequency dampening and True-Class Amplified Asymmetric KL Penalty to coordinate with active structured sparsity updates under extreme imbalance:
\\begin{{equation}}
    L_{{\\text{{EFL}}}} = \\frac{{1}}{{N}} \\sum_{{n=1}}^N \\omega_n \\mathcal{{F}}(t) \\left[ L_{{\\text{{CE}}}}^{{(n)}} + \\lambda_{{\\text{{KL}}}} \\mu(t) \\Lambda_{{\\text{{asym}}}}^{{(n)}} L_{{\\text{{KL}}}}^{{(n)}} \\right]
\\end{{equation}}
where $\\Lambda_{{\\text{{asym}}}}^{{(n)}} = \\min(\\omega_{{y_n}}, \\Lambda_{{\\text{{max}}}})$ squashes false majority evidence on rare-class samples without triggering catastrophic gradient clipping. 

\\paragraph{{Bias-Corrected Temperature Scaling.}} To counteract prior shift from long-tailed re-weighting, a scalar temperature $T$ and a prior-correction bias vector $\\bm{{b}} \\in \\R^K$ are optimized on a validation set: $z^{{\\text{{calib}}}}_c = \\frac{{z^{{\\text{{base}}}}_c}}{{T}} + b_c$. This calibration module is evaluated in future ablations; the current case study uses validation-based thresholding.

\\section{{Experimental Setup: Initial ISIC 2024 Case Study}}\\label{{sec:experiment}}
The current draft reports an initial case study on the ISIC 2024 Challenge dataset~\\cite{{isic2024}} (3D-TBP images with 0.15\\% minority prevalence). We adopt a stratified 70/10/20 train/validation/test split grouped by patient. We note that utilizing the validation set for threshold sweeps represents a limitation, while the 20\\% test partition acts as a local hold-out set.

\\paragraph{{Baselines.}} We compare GUDS-EDL against three Evidential Deep Learning (EDL) baselines: Fisher EDL~\\cite{{fisher_edl2023}}, Flexible EDL~\\cite{{flexible_edl2023}}, and R-EDL~\\cite{{sensoy2018evidential}}. We evaluate models using Sensitivity, Specificity, Macro-AUROC, $\\text{{pAUC}}_{{0.80}}$, Expected Calibration Error (ECE), and Area Under the Risk--Coverage Curve (AURC).

\\section{{Results}}\\label{{sec:results}}
{table1}

{table2}

GUDS-EDL improves ranking and high-recall operating performance, while calibration remains mixed. As shown in Table~\\ref{{tab:operating_modes}}, GUDS-EDL achieves the highest Specificity under the High-Recall Fail-Safe mode, demonstrating the clinical utility of uncertainty-guided sparsity in preserving rare-event representations. Table~\\ref{{tab:threshold_independent}} confirms that GUDS-EDL attains the highest $\\text{{pAUC}}_{{0.80}}$, though ECE suggests that further refinement of the Dirichlet calibration is required.

\\section{{Planned Generalization Protocol}}\\label{{sec:planned_protocol}}
The current empirical evidence is limited to the ISIC 2024 case study. To test whether the proposed topology adaptation generalizes beyond medical rare-event screening, we define a planned protocol on CIFAR-100-LT for controlled long-tailed recognition and MVTec AD for image-level industrial anomaly detection. These planned experiments will compare GUDS-EDL against cross-entropy, logit adjustment, and RigL-style dynamic sparsity, supported by hardware profiling.

\\section{{Conclusion and Limitations}}\\label{{sec:conclusion}}
Through the ISIC 2024 case study, we provide initial evidence that uncertainty-guided sparse topology adaptation can improve rare-event ranking and high-recall operating specificity under extreme class imbalance. The planned CIFAR-100-LT, MVTec AD, and hardware-efficiency protocols define the next stage for validating generality beyond the medical domain. A current limitation is the reliance on the validation partition for threshold tuning.

{bibliography}

\\appendix
\\section{{Supplementary Material}}

\\subsection{{Reproducibility Checklist and Implementation Details}}
To support reproducibility, we report the following implementation details:
\\begin{{itemize}}
    \\item \\textbf{{Hardware:}} All models are trained on NVIDIA RTX A6000 GPUs using 16-bit mixed precision.
    \\item \\textbf{{Hyperparameters:}} The base learning rate is set to $4 \\times 10^{{-5}}$ with an AdamW optimizer, warming up linearly over the first 12 epochs. The total number of training epochs is 40.
    \\item \\textbf{{Topology Variables:}} The structural update rate is $\\eta_{{\\text{{struct}}}} = 0.02$ and the EMA momentum is $\\beta_m = 0.95$.
    \\item \\textbf{{Asymmetric Regularization:}} The KL penalty is $\\lambda_{{\\text{{KL}}}} = 0.1$, capped at a maximum asymmetric multiplier of $\\Lambda_{{\\text{{max}}}} = 10.0$.
\\end{{itemize}}

\\subsection{{Extended Mathematical Derivations}}
The full probability density function (PDF) of the Dirichlet distribution for a probability vector $\\bm{{p}} = [p_1, \\dots, p_K]^T$ on the $K$-dimensional simplex $\\Delta^K$ is defined as:
\\begin{{equation}}\\label{{eq:dirichlet_pdf}}
    \\Dir(\\bm{{p}} \\mid \\bm{{\\alpha}}) = \\frac{{1}}{{B(\\bm{{\\alpha}})}} \\prod_{{c=1}}^K p_c^{{\\alpha_c - 1}} = \\frac{{\\Gamma(S)}}{{\\prod_{{c=1}}^K \\Gamma(\\alpha_c)}} \\prod_{{c=1}}^K p_c^{{\\alpha_c - 1}}
\\end{{equation}}

\\begin{{remark}}[Why not use ReLU?]
ReLU truncates negative logits to zero, resulting in zero gradients for those activations and permanently trapping evidence at $e_c = 0$, which freezes the Dirichlet state on the zero-evidence simplex.
\\end{{remark}}

Under Subjective Logic (SL), a multinomial opinion maps this evidence to belief masses $b_c \\ge 0$ and a vacant belief mass (uncertainty) $u_e \\ge 0$. The belief mass for class $c$ is defined as $b_c = \\frac{{e_c}}{{S}}$. The core identity of Subjective Logic guarantees that the sum of belief masses and epistemic uncertainty equals unity:
\\begin{{equation}}\\label{{eq:sl_identity}}
    \\sum_{{c=1}}^K b_c + u_e = 1
\\end{{equation}}
which establishes the strict mathematical coupling between the evidence vector $\\bm{{e}}$, class belief representations, and the vacant belief mass.

\\subsection{{Extended Algorithm}}
{algorithm1}

\\subsection{{Planned Evaluation Protocol Tables}}
The tables below define our planned evaluation protocol for future multi-domain validation. Note that empirical values remain to be determined (TBD) upon completion of the respective benchmarking sweeps.

{table3}

{table4}

{table5}

{table6}

{table7}

\\end{{document}}
"""

with open(r'd:\MDEP\main_text_new.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)
