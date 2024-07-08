# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

ARG JLVERSION=1.10.4
ARG PYVERSION=3.12.2
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE}

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# Install font used for GUI
RUN mkdir -p /usr/local/share/fonts/IBM_Plex_Mono && \
    curl -L https://github.com/IBM/plex/releases/download/v6.4.0/IBM-Plex-Mono.zip -o /tmp/IBM_Plex_Mono.zip && \
    unzip /tmp/IBM_Plex_Mono.zip -d /usr/local/share/fonts/IBM_Plex_Mono && \
    rm /tmp/IBM_Plex_Mono.zip
RUN fc-cache -f -v

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

RUN python -m venv $HOME/.venv

ENV PYTHON="${HOME}/.venv/bin/python"
ENV PIP="${PYTHON} -m pip"
ENV PATH="${HOME}/.venv/bin:${PATH}"

WORKDIR $HOME/pysr

# Install all requirements, and then PySR itself
COPY --chown=user ./requirements.txt $HOME/pysr/requirements.txt
RUN $PIP install --no-cache-dir -r $HOME/pysr/requirements.txt

COPY --chown=user ./gui/requirements.txt $HOME/pysr/gui/requirements.txt
RUN $PIP install --no-cache-dir -r $HOME/pysr/gui/requirements.txt

COPY --chown=user ./pyproject.toml $HOME/pysr/pyproject.toml
COPY --chown=user ./setup.py $HOME/pysr/setup.py
COPY --chown=user ./pysr $HOME/pysr/pysr
RUN $PIP install --no-cache-dir .

# Install Julia pre-requisites:
RUN $PYTHON -c 'import pysr'

COPY --chown=user ./gui/*.py $HOME/pysr/gui/

EXPOSE 7860
ENV GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

# metainformation
LABEL org.opencontainers.image.authors = "Miles Cranmer"
LABEL org.opencontainers.image.source = "https://github.com/MilesCranmer/PySR"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

CMD ["/home/user/.venv/bin/python", "/home/user/pysr/gui/app.py"]
