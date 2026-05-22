FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Initialize DVC
RUN dvc init --no-scm 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DVC_NO_ANALYTICS=1

# Keep container alive
CMD ["tail", "-f", "/dev/null"]
