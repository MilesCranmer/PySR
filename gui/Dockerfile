FROM julia:1.10.0 AS jl
FROM python:3.12

COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libpython3-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
WORKDIR /home/user/
# Set home to the user's home directory
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

RUN python -m venv /home/user/.venv

# Install Python dependencies in a virtual environment
RUN /home/user/.venv/bin/python -m pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install and pre-compile Julia dependencies,
# including the Bumper extension
RUN /home/user/.venv/bin/python -c "import pysr"
RUN /home/user/.venv/bin/python -c "import pysr; pysr.PySRRegressor(bumper=True, verbosity=0, progress=False, max_evals=1).fit([[1]], [1])"

WORKDIR /home/user/app
COPY --chown=user . $HOME/app

EXPOSE 7860
ENV GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

CMD ["/bin/bash", "-l", "-c", "/home/user/.venv/bin/python /home/user/app/app.py"]
