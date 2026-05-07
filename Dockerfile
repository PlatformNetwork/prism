FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker-cli \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[lium]" && pip install --no-cache-dir --no-deps lium.io==0.0.11
RUN useradd --create-home --shell /usr/sbin/nologin prism \
    && mkdir -p /data \
    && chown -R prism:prism /app /data
USER prism
ENV HOME=/home/prism
EXPOSE 8080
CMD ["uvicorn", "prism_challenge.app:app", "--host", "0.0.0.0", "--port", "8080"]
