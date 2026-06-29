import sys
from isic_paper_experiments import main

if __name__ == "__main__":
    experiments = [
        "full_guds", "guds_without_pruner", "guds_without_regrower", 
        "guds_asymmetric_kl", "guds_without_efl", "guds_without_anticryst",
        "guds_absolute_pruner", "guds_class_conditioned_regrower",
        "guds_without_topology_cache", "guds_temperature_only",
        "guds_no_posthoc_calibration"
    ]
    
    # Check if --experiments is already passed
    if not any(arg.startswith("--experiments") for arg in sys.argv):
        sys.argv.extend(["--experiments"] + experiments)
    
    print(f"Running ISIC GUDS-EDL Main and Ablations: {experiments}")
    main()
