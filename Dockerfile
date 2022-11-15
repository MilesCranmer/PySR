# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG PYVERSION=3.10.8

FROM python:$PYVERSION

# metainformation
LABEL org.opencontainers.image.authors = "Miles Cranmer"
LABEL org.opencontainers.image.source = "https://github.com/MilesCranmer/PySR"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

# Need to use ARG after FROM, otherwise it won't get passed through.
ARG JLVERSION=1.8.2

RUN apt-get update && apt-get install -yq \
        expect \
        build-essential \
        curl \
        git \
        libssl-dev \
        libffi-dev \
        libxml2 \
        libxml2-dev \
        libxslt1.1 \
        libxslt-dev \
        libz-dev \
        nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install juliaup:
RUN curl -fsSL https://install.julialang.org -o install-julia.sh && \
    # Fix for docker buildx https://github.com/rust-lang/rustup/issues/2700
    sed -i 's#/proc/self/exe#$(which head)#g' install-julia.sh && \
    sed -i 's#/proc/cpuinfo#/proc/cpuinfo 2> /dev/null || echo ''#g' install-julia.sh && \
    sed -i 's#get_architecture || return 1#RETVAL=$(gcc -dumpmachine | sed "s/-/-unknown-/") #g' install-julia.sh && \
    # Fix for non-interactivity https://github.com/JuliaLang/juliaup/issues/253
    echo '#!/usr/bin/expect\nspawn ./install-julia.sh\nexpect "Cancel installation"\nsend -- "\\r"\nexpect eof' >> install-julia.exp && \
    chmod +x install-julia.sh && \
    chmod +x install-julia.exp && \
    ./install-julia.exp && \
    rm install-julia.sh && \
    rm install-julia.exp

ENV JULIAUP_ROOT /root/.julia/juliaup
ENV PATH "${JULIAUP_ROOT}/bin:${PATH}"
RUN juliaup add $JLVERSION && juliaup default $JLVERSION && juliaup update

# Install IPython and other useful libraries:
RUN pip install ipython matplotlib

WORKDIR /pysr

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

CMD ["ipython"]
