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
    
    # Inject --experiment flags if not already passed by user
    if not any(arg.startswith("--experiment") for arg in sys.argv[1:]):
        for exp in experiments:
            sys.argv.extend(["--experiment", exp])
    
    print(f"Running ISIC GUDS-EDL Main and Ablations: {experiments}")
    main()
