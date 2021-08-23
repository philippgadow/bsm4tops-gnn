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
    echo "Activating conda environment."
    source conda/bin/activate
    conda activate bsm4tops-gnn
else 
    echo "Setting up conda environment..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p conda
    rm Miniconda3-py39_4.10.3-Linux-x86_64.sh
    source conda/bin/activate
    conda create --name bsm4tops-gnn -y
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    conda install pytorch-geometric -c rusty1s -c conda-forge -y
    conda install uproot vector matplotlib seaborn mplhep -c conda-forge -y

    # python3 -m venv venv
    # source venv/bin/activate
    # pip3 install -r requirements.txt
    # export CUDA=cpu

    # pip3 install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    # pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    # pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    # pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    # pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
    # pip3 install torch-geometric
fi
