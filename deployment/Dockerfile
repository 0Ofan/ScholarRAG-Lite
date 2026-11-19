FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/app/.cache/pip \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt
COPY deployment/requirements-cpu.txt ./deployment/requirements-cpu.txt
RUN python -m venv /app/.venv \
    && /app/.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r deployment/requirements-cpu.txt
COPY app.py ./
COPY src ./src
COPY prompts ./prompts
COPY experiments/results/phase33/comparison.json ./experiments/results/phase33/comparison.json
COPY experiments/results/phase34_ablation/comparison.json ./experiments/results/phase34_ablation/comparison.json
RUN useradd --uid 10001 --create-home appuser && mkdir -p .cache && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["/app/.venv/bin/python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
