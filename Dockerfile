FROM python:3.10-slim-bookworm

# Install required build tools
RUN apt update -y && \
    apt install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    awscli \
    curl \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements-docker.txt

CMD ["python3", "app.py"]
