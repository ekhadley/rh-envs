FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /agent
RUN mkdir -p /secrets
COPY guess.py guess_noisy.py /opt/
RUN chmod +x /opt/guess.py /opt/guess_noisy.py
