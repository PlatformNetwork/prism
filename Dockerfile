FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[lium]" && pip install --no-cache-dir --no-deps lium.io==0.0.11
EXPOSE 8080
CMD ["uvicorn", "prism_challenge.app:app", "--host", "0.0.0.0", "--port", "8080"]
