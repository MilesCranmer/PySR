FROM python:3.9

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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

# Install Julia:
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel=1.10.0
# Install Julia dependencies
RUN python -c "import pysr"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
