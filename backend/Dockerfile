FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOOSRA_DB=/data/doosra.sqlite \
    DOOSRA_MODELS_DIR=/data/models
# On Cloud Run set DOOSRA_DB to the Neon postgresql:// URL: the prediction log, bets and
# models then live in the database, and /data is only a scratch copy of the models.

WORKDIR /app
COPY pyproject.toml ./
COPY doosra ./doosra
RUN pip install --no-cache-dir ".[llm,cloud]" && useradd --create-home doosra && mkdir -p /data && chown doosra /data
USER doosra

EXPOSE 8000
CMD ["sh", "-c", "uvicorn doosra.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
