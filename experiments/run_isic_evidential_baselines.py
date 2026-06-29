import sys
from isic_paper_experiments import main

if __name__ == "__main__":
    experiments = [
        "dense_edl", "fisher_edl", "flexible_edl", 
        "r_edl", "static_24_edl", "rigl_style_24"
    ]
    
    # Inject --experiment flags if not already passed by user
    if not any(arg.startswith("--experiment") for arg in sys.argv[1:]):
        for exp in experiments:
            sys.argv.extend(["--experiment", exp])
    
    print(f"Running ISIC Evidential Baselines: {experiments}")
    main()
