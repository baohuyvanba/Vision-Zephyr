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

# Initial Conda Environment
RUN conda create -n geni-llava python=3.10 -y && \
    conda clean -afy

ENV PATH="/usr/local/cuda-12.9/bin:$PATH"
# ENV LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}"

RUN git clone https://username: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@github.com/baohuyvanba/geni.git /home/vscode/Geni && \
    conda run -n geni-llava pip install transformers torch pillow datasets accelerate && \
    conda run -n geni-llava pip install flash-attn --no-build-isolation