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

RUN apt-get update && apt-get install -y expect && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install juliaup:
RUN curl -fsSL https://install.julialang.org -o /tmp/install-julia.sh && \
    echo '#!/usr/bin/expect\nspawn /tmp/install-julia.sh\nexpect "Cancel installation"\nsend -- "\\r"\nexpect eof' >> /tmp/install-julia.exp && \
    chmod +x /tmp/install-julia.sh && \
    chmod +x /tmp/install-julia.exp && \
    /tmp/install-julia.exp && \
    rm /tmp/install-julia.sh && \
    rm /tmp/install-julia.exp

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
