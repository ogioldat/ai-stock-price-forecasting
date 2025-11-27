FROM python:3.13-slim AS builder

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv
RUN uv pip install -e . --system

COPY api/src/ ./src/

FROM python:3.13-slim AS runner

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder app/src ./src

ENV PYTHONUNBUFFERED 1

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
