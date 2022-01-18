# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG VERSION=latest

FROM julia:$VERSION

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    vim git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pysr

# Install PyEnv to switch Python to dynamically linked version:
RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:$PATH"

ENV PYTHON_VERSION="3.9.10"
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PYTHON_VERSION}
ENV PATH="/root/.pyenv/versions/${PYTHON_VERSION}/bin:$PATH"

# Install IPython and other useful libraries:
RUN pip install ipython jupyter matplotlib

# Caches install (https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project)
ADD ./requirements.txt /pysr/requirements.txt
RUN pip3 install -r /pysr/requirements.txt

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./setup.py /pysr/setup.py
ADD ./README.md /pysr/README.md
Add ./Project.toml /pysr/Project.toml
ADD ./pysr/ /pysr/pysr/
RUN pip3 install .

# Install Julia pre-requisites:
RUN python3 -c 'import pysr; pysr.install()'

CMD ["bash"]