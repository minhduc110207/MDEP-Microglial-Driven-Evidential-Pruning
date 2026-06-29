import sys
from isic_paper_experiments import main

if __name__ == "__main__":
    experiments = [
        "standard_ce", "focal_loss", 
        "class_balanced_ce", "balanced_softmax", "ldam_drw", 
        "decoupled_crt", "mislas"
    ]
    
    # Inject --experiment flags if not already passed by user
    if not any(arg.startswith("--experiment") for arg in sys.argv[1:]):
        for exp in experiments:
            sys.argv.extend(["--experiment", exp])
    
    print(f"Running ISIC Softmax Baselines: {experiments}")
    main()
