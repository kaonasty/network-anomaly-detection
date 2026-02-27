FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Generate data and features at build time (optional — can mount volumes instead)
# RUN python data/generate_data.py && python features/feature_engineering.py

# Expose serving port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD curl -f http://localhost:8080/health || exit 1

# Run serving
CMD ["uvicorn", "serving.serve:app", "--host", "0.0.0.0", "--port", "8080"]
