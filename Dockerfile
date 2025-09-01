# Use a small, multi-arch Python image (works on Intel & Apple Silicon)
FROM python:3.11-slim

# Install CBC solver (PuLP will use this)
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy and install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your source (includes templates/)
COPY . .

# Expose the port we'll listen on
ENV PORT=8080
# Optional: ensure Flask doesn’t try to use reloader in prod
ENV PYTHONUNBUFFERED=1

# Start with Gunicorn (2 workers, threaded)
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "120", "-b", "0.0.0.0:8080", "app:app"]

