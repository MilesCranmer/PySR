FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies:
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install Julia:
RUN python -c "import pysr"

COPY . .

CMD ["python", "app.py"]
