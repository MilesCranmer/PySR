# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG JLVERSION=1.9.4
ARG PYVERSION=3.11.6
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE}

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# Install IPython and other useful libraries:
RUN pip install --no-cache-dir ipython matplotlib

WORKDIR /pysr

# Caches install (https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project)
ADD ./requirements.txt /pysr/requirements.txt
RUN pip3 install --no-cache-dir -r /pysr/requirements.txt

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./pyproject.toml /pysr/pyproject.toml
ADD ./setup.py /pysr/setup.py
ADD ./pysr /pysr/pysr
RUN pip3 install --no-cache-dir .

# Install Julia pre-requisites:
RUN python3 -c 'import pysr'

# metainformation
LABEL org.opencontainers.image.authors = "Miles Cranmer"
LABEL org.opencontainers.image.source = "https://github.com/MilesCranmer/PySR"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

CMD ["ipython"]
