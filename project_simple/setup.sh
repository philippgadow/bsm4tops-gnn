#!/bin/bash

if [ -f "/singularity" ]; then
    echo "You are inside a singularity container. There is no need to call setup.sh, everything is already installed."
    return 0
fi

if [ -f "/.dockerenv" ]; then
    echo "You are inside a Docker container. There is no need to call setup.sh, everything is already installed."
    return 0
fi

if [ -d "conda" ]; then
    echo "Activating conda environment."
    source conda/bin/activate
    conda activate bsm4tops
else 
    echo "Setting up conda environment..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p conda
    rm Miniconda3-py39_4.10.3-Linux-x86_64.sh
    source conda/bin/activate
    conda env create -f environment.yml
    conda activate bsm4tops
fi
