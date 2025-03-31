#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Python 3.11 development setup on EC2..."

# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# Install necessary dependencies for adding new PPA
sudo apt install -y software-properties-common

# Add deadsnakes PPA for Python 3.11 (if not already available)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11 and related tools
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Ensure python3 points to Python 3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --config python3

# Verify installation
python3 --version
pip3 --version

# Install virtual environment tools
pip3 install --upgrade virtualenv wheel setuptools

# Install additional development tools
sudo apt install -y build-essential git curl unzip htop

# Optionally set up swap space (recommended for low-memory instances)
SWAP_SIZE="4G"
SWAP_FILE="/swapfile"

if [ ! -f "$SWAP_FILE" ]; then
  echo "Setting up $SWAP_SIZE swap space..."
  sudo fallocate -l $SWAP_SIZE $SWAP_FILE
  sudo chmod 600 $SWAP_FILE
  sudo mkswap $SWAP_FILE
  sudo swapon $SWAP_FILE
  echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab
else
  echo "Swap file already exists, skipping swap setup."
fi

# Create a Python virtual environment (optional)
python3 -m venv ~/venv

source venv/bin/activate

git clone https://github.com/aerithnetzer/anthropocene-bertopic.git

cd anthropocene-bertopic

pip3 install -r requirements.txt

aws s3 --recursive s3://anthropocene-data/constellate/ . --include "*.txt" --exclude "*"

echo "Python 3.11 development setup complete!"
echo "Activate your virtual environment with: source $DEV_DIR/venv/bin/activate"
