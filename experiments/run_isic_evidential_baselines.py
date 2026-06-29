import sys
from isic_paper_experiments import main

if __name__ == "__main__":
    experiments = [
        "dense_edl", "fisher_edl", "flexible_edl", 
        "r_edl", "static_24_edl", "rigl_style_24"
    ]
    
    # Check if --experiments is already passed
    if not any(arg.startswith("--experiments") for arg in sys.argv):
        sys.argv.extend(["--experiments"] + experiments)
    
    print(f"Running ISIC Evidential Baselines: {experiments}")
    main()
