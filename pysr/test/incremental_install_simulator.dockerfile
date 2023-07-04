# This dockerfile simulates a user installation that first
# builds PySR for Python 3.9, and then upgrades to Python 3.10.
# Normally this would cause an error when installing PyCall, so we want to
# ensure that PySR can automatically patch things.
FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install juliaup and pyenv:
RUN apt-get update && apt-get install -y curl git build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install juliaup:
RUN curl -fsSL https://install.julialang.org | sh -s -- -y

# Install pyenv:
RUN curl -fsSL curl https://pyenv.run | sh && \
    echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Default to using bash -l:
SHELL ["/bin/bash", "-l", "-c"]

RUN juliaup add 1.8 && juliaup default 1.8
RUN pyenv install 3.9.2 && pyenv global 3.9.2
RUN python3 -m pip install --upgrade pip

# Get PySR source:
WORKDIR /pysr
ADD ./requirements.txt /pysr/requirements.txt
RUN python3 -m pip install -r /pysr/requirements.txt

ADD ./setup.py /pysr/setup.py
ADD ./pysr/ /pysr/pysr/

# First install of PySR:
RUN python3 -m pip install .
RUN python3 -m pysr install

# Change Python version:
RUN pyenv install 3.10 && pyenv global 3.10 && pyenv uninstall -f 3.9.2
RUN python3 -m pip install --upgrade pip

# Second install of PySR:
RUN python3 -m pip install .
RUN rm -r ~/.julia/environments/pysr-*
RUN python3 -m pysr install
