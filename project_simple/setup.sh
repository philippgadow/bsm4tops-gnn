#!/bin/bash

if [ -f "/singularity" ]; then
    echo "You are inside a singularity container. There is no need to call setup.sh, everything is already installed."
    return 0
fi

if [ -f "/.dockerenv" ]; then
    echo "You are inside a Docker container. There is no need to call setup.sh, everything is already installed."
    return 0
fi

if [ -d "venv" ]; then
    echo "Activating virtual environment."
    source venv/bin/activate
else 
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    export CUDA=cpu

    pip3 install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    pip3 install torch-geometric
fi
