#!/bin/bash
# CI Bootstrap Script - Install dependencies and run proof verification
set -e

echo " Starting CI Proof Pipeline..."

# Install Python dependencies
echo " Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install z3-solver requests

# Run proof verification
echo " Running proof verification..."
python -m hermes.ci.cli local

echo " CI Proof Pipeline completed successfully!"