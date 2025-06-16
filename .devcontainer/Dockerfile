FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

SHELL [ "bash", "-c" ]

# Ubuntu Mirror server
ARG UBUNTU_MIRROR=archive.ubuntu.com/ubuntu

# Update cache, turn on universe, install system packages
RUN apt-get update --fix-missing && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -yq \
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

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2004-12-9-local_12.9.0-575.51.03-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-12-9-local_12.9.0-575.51.03-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2004-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-9

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -yq git-lfs && \
    git lfs install

############################################
# Setup user
############################################
USER vscode
WORKDIR /home/vscode

# Miniconda
ENV CONDA_DIR=/home/vscode/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p "$CONDA_DIR" && \
    rm miniconda.sh && \
    conda init --all

ENV PATH="/usr/local/cuda-12.9/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}"

# Git Clone & Environment initialization
RUN git clone https://username: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@github.com/baohuyvanba/Vision-Zephyr.git /home/vscode/VisZephyr &&\
    cd /home/vscode/VisZephyr && \
    conda env create -f environment.yaml -n viszephyr && \
    conda clean -afy

# Data
# RUN mkdir -p /home/vscode/VisZephyr/playground/data/Pretrain && \
#     cd /home/vscode/VisZephyr/playground/data/Pretrain && \
#     wget -c https://huggingface.co/datasets/liuhaotian/Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json && \
#     wget -c https://huggingface.co/datasets/liuhaotian/Pretrain/resolve/main/blip_laion_cc_sbu_558k.json && \
#     wget -c https://huggingface.co/datasets/liuhaotian/Pretrain/resolve/main/images.zip