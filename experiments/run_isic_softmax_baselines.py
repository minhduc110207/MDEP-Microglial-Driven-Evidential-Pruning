import sys
from isic_paper_experiments import main

if __name__ == "__main__":
    experiments = [
        "standard_ce", "focal_loss", "logit_adjustment", 
        "class_balanced_ce", "balanced_softmax", "ldam_drw", 
        "decoupled_crt", "mislas"
    ]
    
    # Check if --experiments is already passed
    if not any(arg.startswith("--experiments") for arg in sys.argv):
        sys.argv.extend(["--experiments"] + experiments)
    
    print(f"Running ISIC Softmax Baselines: {experiments}")
    main()
