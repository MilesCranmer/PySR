# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG JLVERSION=1.10.0
ARG PYVERSION=3.11.6
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE}

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# Install dependencies for GUI:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libpython3-dev \
        libfreetype6-dev \
        pkg-config \
        libfontconfig1 \
        fontconfig \
        curl \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install IPython and other useful libraries:
RUN pip install --no-cache-dir ipython matplotlib

WORKDIR /pysr

# Caches install (https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project)
ADD ./requirements.txt /pysr/requirements.txt
RUN pip3 install --no-cache-dir -r /pysr/requirements.txt

################################################################################
## GUI setup
# Install IBM Plex Mono font (so our plots look Gradio-style)
RUN mkdir -p /usr/local/share/fonts/IBM_Plex_Mono && \
    curl -L https://github.com/IBM/plex/releases/download/v6.4.0/IBM-Plex-Mono.zip -o /tmp/IBM_Plex_Mono.zip && \
    unzip /tmp/IBM_Plex_Mono.zip -d /usr/local/share/fonts/IBM_Plex_Mono && \
    rm /tmp/IBM_Plex_Mono.zip
RUN fc-cache -f -v
ADD ./gui/requirements.txt /pysr/gui/requirements.txt
RUN pip3 install --no-cache-dir -r /pysr/gui/requirements.txt
################################################################################

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./pyproject.toml /pysr/pyproject.toml
ADD ./setup.py /pysr/setup.py
ADD ./pysr /pysr/pysr
RUN pip3 install --no-cache-dir .

# Install Julia pre-requisites:
RUN python3 -c 'import pysr'

EXPOSE 7860
ENV GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

ADD ./gui/app.py /pysr/gui/app.py

# metainformation
LABEL org.opencontainers.image.authors = "Miles Cranmer"
LABEL org.opencontainers.image.source = "https://github.com/MilesCranmer/PySR"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

CMD ["python3", "/pysr/gui/app.py"]
