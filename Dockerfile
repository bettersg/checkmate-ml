FROM python:3.10.6-slim

WORKDIR /app

# Update package list and install libgl1.
RUN apt-get update && apt-get install\
    libgl1\ 
    libglib2.0-0 -y

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt /app

# Ignore warning on running pip as root user
ENV PIP_ROOT_USER_ACTION=ignore

# Install dependencies.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY . /app

# Copy models downloaded from Cloud Storage into the container image
COPY /files /app/files

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8001}