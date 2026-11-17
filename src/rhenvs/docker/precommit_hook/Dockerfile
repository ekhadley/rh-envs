FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mypy==1.20.2" "types-PyYAML==6.0.12.20260518" "types-requests==2.33.0.20260518" pyyaml "requests>=2.28"
RUN mkdir -p /work
WORKDIR /work
