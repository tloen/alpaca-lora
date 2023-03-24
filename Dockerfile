FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# enter the base model weights you wish to use here, like "decapoda-research/llama-7b-hf"
ENV BASE_MODEL="None"

# since bytesands will use the installed cuda version, I have fixed to 11.8 and cannot use easily torch2.0 or nvidia with pytorch containers
RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install -r requirements.txt \
    && python3.10 -m pip install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
COPY . .
ENTRYPOINT [ "python3.10"]