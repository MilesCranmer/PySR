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

ENV PYVERSION $PYVERSION
ENV JLVERSION $JLVERSION

# arm64:
# https://julialang-s3.julialang.org/bin/linux/aarch64/1.8/julia-1.8.2-linux-aarch64.tar.gz
# amd64:
# https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz

RUN export JULIA_VER=$(echo $JLVERSION | cut -d '.' -f -2) && \
    export ARCH=$(arch | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') && \
    if [ "$ARCH" = "amd64" ]; then \
        export BASE_URL="https://julialang-s3.julialang.org/bin/linux/x64/$JULIA_VER" && \
        export FULL_URL=$BASE_URL/julia-$JLVERSION-linux-x86_64.tar.gz; \
    elif [ "$ARCH" = "arm64" ]; then \
        export BASE_URL="https://julialang-s3.julialang.org/bin/linux/aarch64/$JULIA_VER"; \
        export FULL_URL=$BASE_URL/julia-$JLVERSION-linux-aarch64.tar.gz; \
    else \
        echo "Download link for architecture ${ARCH} not found. Please add the corresponding Julia download URL to this Dockerfile." && \
        exit 1; \
    fi && \
    wget -nv $FULL_URL -O julia.tar.gz && \
    tar -xzf julia.tar.gz && \
    rm julia.tar.gz && \
    mv julia-$JLVERSION /opt/julia && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia

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

# Install Julia pre-requisites:
RUN python3 -c 'import pysr; pysr.install()'

CMD ["ipython"]
