FROM python:3.11

WORKDIR /gaurdrail-app
RUN apt-get update && apt-get install -y stress-ng && rm -rf /var/lib/apt/lists/*

COPY docker-req.txt .
RUN pip install -r docker-req.txt
COPY . .
