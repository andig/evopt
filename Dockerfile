FROM python:3.13-slim

WORKDIR /app

# Install system dependencies including CBC solver for PuLP
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Expose the port
EXPOSE 7050

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7050", "--workers", "4", "app:app"]