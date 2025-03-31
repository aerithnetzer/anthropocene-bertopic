#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Update package list and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common

# Add the deadsnakes PPA for Python 3.11
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11 and dependencies
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils python3-pip

# Ensure Python 3.11 is the default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --config python3

# Verify installation
python3 --version

python3 -m venv ./venv

source venv/bin/activate

pip install -r requirements.txt
