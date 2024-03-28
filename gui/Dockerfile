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

# Install Python dependencies:
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install Julia dependencies
RUN python -c "import pysr"

COPY . .

EXPOSE 7860

CMD ["python", "-X", "juliacall-home=/usr/local/julia/bin", "app.py"]
