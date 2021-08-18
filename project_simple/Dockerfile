FROM continuumio/anaconda3:2021.05

# install basic programs
RUN apt-get update &&  apt-get install -y \
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# create a working directory
RUN mkdir /app
COPY . /app
WORKDIR /app

# install python packages
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch --yes
RUN conda install pytorch-geometric -c rusty1s -c conda-forge --yes
RUN pip3 install -r requirements.txt
