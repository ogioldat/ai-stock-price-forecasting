FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

# Install Node
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs
RUN npm install -g @commitlint/cli @commitlint/config-conventional pnpm

# Python tooling
RUN pip install uv ruff black mypy pytest pre-commit