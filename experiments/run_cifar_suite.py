import sys
from generalization_paper_suite import main

if __name__ == "__main__":
    # Check if --benchmark is already passed
    if not any(arg.startswith("--benchmark") for arg in sys.argv):
        sys.argv.extend(["--benchmark", "cifar"])
    
    print("Running CIFAR-100-LT Generalization Suite")
    main()
