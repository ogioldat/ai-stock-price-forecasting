FROM mcr.microsoft.com/devcontainers/python:3.13-bullseye

# Install Node
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs
RUN npm install -g @commitlint/cli @commitlint/config-conventional pnpm

RUN curl -sSL https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks-linux-amd64 \
    -o /usr/local/bin/gitleaks && chmod +x /usr/local/bin/gitleaks

# Python tooling
RUN pip install uv ruff black mypy pytest pre-commit