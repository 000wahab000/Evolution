# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files and keeps stdout/stderr unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy

WORKDIR /app

# Create a non-privileged user with a valid home directory.
ARG UID=10001
RUN useradd -m -u "${UID}" -s /bin/bash appuser

# Install dependencies using BuildKit pip cache.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt

# Pre-create runtime directories and set ownership.
RUN mkdir -p /app/logs /app/models && chown -R appuser:appuser /app

# Copy application code into container.
COPY --chown=appuser:appuser . .

# Switch to non-privileged user.
USER appuser

# Expose server port.
EXPOSE 8000

# Run with Gunicorn on $PORT (Render sets it; defaults to 8000) using threads for SSE streaming and disable unused control socket.
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 8 --timeout 120 --no-control-socket"]
