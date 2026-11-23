# Образ API асистента ІТ-підтримки. Збирається з кореня проєкту:
#   docker build -f 14/deploy/Dockerfile -t pcdiag-api:1.0 .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf LLM_CACHE=0

WORKDIR /srv
COPY 14/deploy/requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
 && pip install -r /tmp/requirements.txt \
 # модель embeddings завантажується під час збирання — контейнер стартує без доступу до Інтернету
 && python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# код і дані (без .env, кешів і звітів — див. .dockerignore)
COPY pcdiag/ pcdiag/
COPY 8/pcdiag.ttl 8/pcdiag.ttl
COPY 11/index/structural/ 11/index/structural/
COPY 12/app/ 12/app/

# непривілейований користувач
RUN useradd --uid 10001 --create-home appuser && chown -R appuser /srv /opt/hf
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
WORKDIR /srv/12
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
