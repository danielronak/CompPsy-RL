#!/usr/bin/env python3
"""
Convenience launcher for the ChoiceRoom (MiniGrid-style) Choice-Overload
experiment under CNN function approximation.

    python run_minigrid.py --smoke                    # quick CPU sanity check
    python run_minigrid.py --seeds 20 --episodes 1500 # full run (use a GPU)

Uses CUDA automatically when available (e.g. a laptop RTX 3050), else CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.experiments.exp_minigrid_choice import main

if __name__ == "__main__":
    main()
