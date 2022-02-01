# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG ARCH=linux/amd64
ARG VERSION=latest

FROM --platform=$ARCH julia:$VERSION

# Need to use ARG after FROM, otherwise it won't get passed through.
ARG PYVERSION=3.9.10

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    vim git tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pysr

# Install PyEnv to switch Python to dynamically linked version:
RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:$PATH"

# ENV PYVERSION=${PYVERSION}
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PYVERSION}
ENV PATH="/root/.pyenv/versions/$PYVERSION/bin:$PATH"

# Install IPython and other useful libraries:
RUN pip install ipython jupyter matplotlib

# Caches install (https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project)
ADD ./requirements.txt /pysr/requirements.txt
RUN pip3 install -r /pysr/requirements.txt

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./setup.py /pysr/setup.py
ADD ./pysr/ /pysr/pysr/
RUN pip3 install .

# Install Julia pre-requisites:
RUN python3 -c 'import pysr; pysr.install()'

# Add tests
ADD ./test/ /pysr/test/

CMD ["bash"]
