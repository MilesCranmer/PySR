FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies:
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install Julia:
RUN python -c "import pysr"

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["python", "app.py"]
