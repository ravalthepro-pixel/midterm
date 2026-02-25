"""
SETUP SCRIPT — Run this FIRST before train.py
It installs all required Python packages automatically.

How to run:
  1. Open this folder in VS Code
  2. Open the terminal (Terminal → New Terminal)
  3. Type:  python setup.py
  4. Wait for it to finish
  5. Then run:  python train.py
"""

import subprocess
import sys

packages = [
    "torch",
    "torchvision",
    "matplotlib",
    "scikit-learn",
    "seaborn",
    "Pillow",
]

print("=" * 50)
print("Installing required packages...")
print("=" * 50)

for package in packages:
    print(f"\nInstalling {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\n" + "=" * 50)
print("All packages installed successfully!")
print("You can now run:  python train.py")
print("=" * 50)
