# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Run Pygame in dummy/headless mode without X11 display.
ENV SDL_VIDEODRIVER=dummy

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install CPU-only torch before installing other dependencies.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Download remaining dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container with appropriate ownership.
COPY --chown=appuser:appuser . .

# Ensure storage directories exist with non-root ownership.
RUN mkdir -p logs models && chown -R appuser:appuser /app

# Set HOME to a directory appuser can actually write to.
# The appuser was created with --home /nonexistent --no-create-home,
# so $HOME must be overridden or gunicorn's control socket fails.
ENV HOME=/app

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
