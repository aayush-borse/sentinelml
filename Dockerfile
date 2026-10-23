FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY artifacts ./artifacts
RUN pip install --upgrade pip && pip install .
EXPOSE 8000
CMD ["sh", "-c", "python -m sentinelml.cli train && uvicorn sentinelml.api:app --host 0.0.0.0 --port 8000"]
