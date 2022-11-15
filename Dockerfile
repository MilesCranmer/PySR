# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG JLVERSION=latest

FROM julia:$JLVERSION

# metainformation
LABEL org.opencontainers.image.authors = "Miles Cranmer"
LABEL org.opencontainers.image.source = "https://github.com/MilesCranmer/PySR"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

# Need to use ARG after FROM, otherwise it won't get passed through.
ARG PYVERSION=3.10.8

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    vim git tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pysr

# Install PyEnv to switch Python to dynamically linked version:
ENV PYVERSION ${PYVERSION}
ENV PYENV_ROOT /root/.pyenv
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$PATH"
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PYVERSION} \
    && pyenv global ${PYVERSION} \
    && pyenv rehash

# Install IPython and other useful libraries:
RUN pip install ipython matplotlib

# Caches install (https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project)
ADD ./requirements.txt /pysr/requirements.txt
RUN pip3 install -r /pysr/requirements.txt

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./setup.py /pysr/setup.py
ADD ./pysr/ /pysr/pysr/
RUN pip3 install .

# Load version information from the package:
LABEL org.opencontainers.image.version = $(python -c "import pysr; print(pysr.__version__)")

# Install Julia pre-requisites:
RUN python3 -c 'import pysr; pysr.install()'

CMD ["bash"]
