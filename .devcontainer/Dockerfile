FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

SHELL [ "bash", "-c" ]

# Ubuntu Mirror server
# ARG UBUNTU_MIRROR=archive.ubuntu.com/ubuntu
# RUN sed -i "s|http://archive.ubuntu.com/ubuntu|http://$UBUNTU_MIRROR|g" /etc/apt/sources.list

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Update cache, turn on universe, install system packages
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository --yes universe && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
        ffmpeg \
        dkms \
        build-essential \
        jq \
        jp \
        tree \
        tldr \
        git \
        wget \
        curl \
        ca-certificates

#====================================================================================================================================
# Setup user
#====================================================================================================================================
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode
USER vscode
WORKDIR /home/vscode

# Miniconda
ENV CONDA_DIR=/home/vscode/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p "$CONDA_DIR" && \
    rm miniconda.sh && \
    conda init --all

# Git Clone & Environment initialization
RUN git clone --branch finetune --single-branch https://github.com/baohuyvanba/Vision-Zephyr.git /home/vscode/Vision-Zephyr &&\
    cd /home/vscode/Vision-Zephyr && \
    conda env create -f environment.yaml -n viszephyr && \
    conda clean -afy && \
    conda run -n viszephyr pip install flash-attn --no-build-isolation && \
    conda run -n viszephyr pip install wandb shapely scipy shortuuid