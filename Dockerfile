# =========================
# Base Image
# =========================
FROM python:3.11-slim

# =========================
# Set working directory
# =========================
WORKDIR /app

# =========================
# Install system dependencies
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Copy project files
# =========================
COPY . /app

# =========================
# Install Python dependencies
# =========================
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# =========================
# Expose FastAPI port
# =========================
EXPOSE 8000

# =========================
# Run FastAPI with uvicorn
# =========================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
