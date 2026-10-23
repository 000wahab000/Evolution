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

# Install CPU-only torch before installing other dependencies.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies using BuildKit pip cache.
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

# Run with Gunicorn using gevent worker for non-blocking SSE streaming.
CMD ["gunicorn", "-w", "1", "-k", "gevent", "--timeout", "120", "--bind", "0.0.0.0:8000", "app:app"]
