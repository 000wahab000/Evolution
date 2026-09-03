"""
Backward-compatibility entrypoint for scripts/plot_logs.py.
Recommended canonical command: python scripts/plot_logs.py
"""
import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.plot_logs import main

if __name__ == "__main__":
    main()
