FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /app/

# Copy the best model into the image so it's ready to use
COPY best_model.pth /app/best_model.pth

EXPOSE 5000

CMD ["python", "app.py"]
