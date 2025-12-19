#!/usr/bin/env python
"""
Quick training status checker
Shows current training progress and metrics
"""

import os
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
log_file = project_root / "results/logs/training.log"
checkpoint_dir = project_root / "results/checkpoints"

print("="*80)
print("TRAINING STATUS CHECK")
print("="*80)

# Check if training is running
if not log_file.exists():
    print("\n⚠️  Training hasn't started yet or log file doesn't exist")
    print(f"Log file: {log_file}")
    exit(1)

# Read last lines of log file
print("\n📋 Latest Training Output:")
print("-"*80)

with open(log_file, 'r') as f:
    lines = f.readlines()
    # Show last 30 lines
    recent_lines = lines[-30:]
    for line in recent_lines:
        print(line.rstrip())

# Check checkpoints
print("\n" + "-"*80)
print("📁 Checkpoint Files:")
if checkpoint_dir.exists():
    files = list(checkpoint_dir.glob("*.pt"))
    if files:
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_mb = f.stat().st_size / (1024*1024)
            mod_time = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  ✓ {f.name}")
            print(f"    Size: {size_mb:.1f} MB")
            print(f"    Last Modified: {mod_time}")
    else:
        print("  No checkpoints yet")
else:
    print("  Checkpoint directory doesn't exist yet")

print("\n" + "="*80)
print("Tip: Run 'tail -f results/logs/training.log' to see live training output")
print("="*80)
