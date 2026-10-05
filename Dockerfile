FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit
COPY data/documents ./data/documents
COPY evaluation ./evaluation

# Bake embedding + reranker models into the image -> fast cold starts
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

RUN useradd -m app && chown -R app /app
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1

# Render/Railway inject $PORT; defaults to 8000 locally
CMD ["sh", "-c", "uvicorn rag_agent.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
