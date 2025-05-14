#!/bin/bash

set -x  # Echo commands

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Install venv module if not already present
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev

# Create virtual environment using Python 3.12
python3.12 -m venv venv