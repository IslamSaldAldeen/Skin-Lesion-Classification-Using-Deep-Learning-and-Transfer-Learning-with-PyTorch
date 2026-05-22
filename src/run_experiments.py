"""Run the full advanced experiment plan.

Tip: start with small epochs to verify the pipeline, then increase epochs.
"""
import subprocess
import sys

EXPERIMENTS = [
    ["--experiment-name", "exp1_resnet50_unweighted_head", "--architecture", "resnet50", "--freeze", "head", "--epochs", "6", "--lr", "1e-3"],
    ["--experiment-name", "exp2_resnet50_weighted_head", "--architecture", "resnet50", "--freeze", "head", "--weighted-loss", "--epochs", "8", "--lr", "1e-3"],
    ["--experiment-name", "exp3_resnet50_weighted_last_block", "--architecture", "resnet50", "--freeze", "last_block", "--weighted-loss", "--epochs", "10", "--lr", "1e-4"],
    ["--experiment-name", "exp4_resnet50_weighted_whole", "--architecture", "resnet50", "--freeze", "whole", "--weighted-loss", "--epochs", "12", "--lr", "1e-5"],
    ["--experiment-name", "exp5_efficientnet_b0_weighted", "--architecture", "efficientnet_b0", "--freeze", "last_block", "--weighted-loss", "--epochs", "10", "--lr", "1e-4"],
]

for args in EXPERIMENTS:
    print("\n" + "=" * 90)
    print("Running:", " ".join(args))
    print("=" * 90)
    subprocess.run([sys.executable, "-m", "src.train", *args], check=True)
