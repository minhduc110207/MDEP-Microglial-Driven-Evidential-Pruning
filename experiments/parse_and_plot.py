import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

log_path = r"d:\MDEP\experiments\tải xuống.txt"
output_dir = r"d:\MDEP\output"
os.makedirs(output_dir, exist_ok=True)

with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parsing logic
train_losses = [] 
eval_metrics = []

current_exp = None
current_seed = None

for line in lines:
    line = line.strip()
    
    # Example: [RUN] dataset=isic experiment=fisher_edl family=evidential_baseline evaluator=evidential seed=42
    run_match = re.search(r'\[RUN\]\s+.*experiment=(\w+)\s+.*seed=(\d+)', line)
    if run_match:
        current_exp = run_match.group(1)
        current_seed = run_match.group(2)
        continue
    
    # Example: [TRAIN] epoch=005/040 loss=0.4018
    train_match = re.search(r'\[TRAIN\]\s+epoch=(\d+)/\d+\s+loss=([\d\.]+)', line)
    if train_match and current_exp:
        epoch = int(train_match.group(1))
        loss = float(train_match.group(2))
        train_losses.append({'experiment': current_exp, 'seed': current_seed, 'epoch': epoch, 'loss': loss})
        
    # Example: pAUC 0.80 (ISIC 2024)      : 0.0347
    metric_match = re.search(r'([\w\s\-\.\(\)]+)\s*:\s*([\d\.]+)', line)
    if metric_match and current_exp:
        metric_name = metric_match.group(1).strip()
        try:
            metric_val = float(metric_match.group(2))
            if metric_name in ['pAUC 0.80 (ISIC 2024)', 'Macro-AUROC', 'ECE (Adaptive, 15 bins)', 'Mean Epistemic uncertainty', 'Mean Aleatoric uncertainty']:
                eval_metrics.append({'experiment': current_exp, 'seed': current_seed, 'metric': metric_name, 'value': metric_val})
        except ValueError:
            pass

    # Example: Balanced Accuracy         |           0.5000 |           0.7323 |           0.5000
    if current_exp and "Balanced Accuracy" in line and "|" in line:
        parts = line.split("|")
        if len(parts) >= 3:
            try:
                bal_acc_opt = float(parts[2].strip())
                eval_metrics.append({'experiment': current_exp, 'seed': current_seed, 'metric': 'Balanced Accuracy (Opt)', 'value': bal_acc_opt})
            except ValueError:
                pass


df_loss = pd.DataFrame(train_losses)
df_metrics = pd.DataFrame(eval_metrics)

# Set style
sns.set_theme(style="whitegrid", palette="muted")

# 1. Training Loss Plot
if not df_loss.empty:
    plt.figure(figsize=(10, 6))
    # Aggregate over seeds
    sns.lineplot(data=df_loss, x='epoch', y='loss', hue='experiment', marker='o')
    plt.title('Training Loss over Epochs', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(title='Experiment')
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 2. Metrics Bar Plots
if not df_metrics.empty:
    for metric in df_metrics['metric'].unique():
        plt.figure(figsize=(8, 5))
        subset = df_metrics[df_metrics['metric'] == metric]
        sns.barplot(data=subset, x='experiment', y='value', capsize=.1, errorbar='sd')
        plt.title(f'Comparison of {metric}', fontsize=16, fontweight='bold')
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Experiment', fontsize=12)
        plt.xticks(rotation=15)
        safe_metric_name = metric.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
        plt.savefig(os.path.join(output_dir, f'{safe_metric_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

print("Plots generated successfully in", output_dir)
