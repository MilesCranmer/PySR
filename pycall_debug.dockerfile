FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install juliaup and pyenv:
RUN apt-get update && apt-get install -y curl

# Install juliaup:
RUN curl -fsSL https://install.julialang.org | sh -s -- -y

RUN apt-get install -y git build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
    xz-utils libffi-dev liblzma-dev

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

# Try to install pysr:
RUN python3 -m pip install pysr==0.14.2
RUN python3 -m pysr install

# Change Python version:
RUN pyenv install 3.10 && pyenv global 3.10 && pyenv uninstall -f 3.9.2
RUN python3 -m pip install --upgrade pip

# Try to use PySR:
RUN python3 -m pip install pysr==0.14.2
RUN rm -r ~/.julia/environments/pysr-0.14.2
# RUN julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall")'
# RUN python3 -m pysr install
