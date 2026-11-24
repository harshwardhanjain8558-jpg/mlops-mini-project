FROM python:3.12-slim

# print logs straight away (no buffering) and skip .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# install libraries first, so this layer is cached when only our code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# then our code, settings and the champion model
COPY params.yaml .
COPY src/ src/
COPY app/ app/
COPY models/ models/

# don't run as root inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# --no-access-log: keep stdout to our JSON lines (no plain-text probe spam)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
