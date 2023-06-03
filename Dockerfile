FROM python:3.10.6-slim

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt ./

# Install dependencies.
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY . ./

CMD uvicorn app:app --host 0.0.0.0 --port $PORT