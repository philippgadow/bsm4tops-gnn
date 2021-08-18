#!/bin/bash

if [ -d "venv" ]; then
    echo "Activating virtual environment."
    source venv/bin/activate
else 
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    export CUDA=cpu
    pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip3 install torch-geometric
fi
