#!/usr/bin/env python3
"""Start MarketPulse: runs the pipeline then launches the dashboard."""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

print("Running pipeline...")
result = subprocess.run([sys.executable, "scripts/run_pipeline.py"], cwd=ROOT)
if result.returncode != 0:
    print("Pipeline failed. Fix errors above before launching the dashboard.")
    sys.exit(1)

print("Launching dashboard at http://localhost:8501 ...")
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/MarketPulse.py"], cwd=ROOT)
