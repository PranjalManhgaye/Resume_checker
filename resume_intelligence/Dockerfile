# WIP — see PR / issue #1
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# TODO: multi-stage, non-root user, healthcheck, choose streamlit vs api entrypoint
CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
