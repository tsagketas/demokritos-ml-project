FROM python:3.11-slim

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set KaggleHub dataset storage
ENV KAGGLEHUB_CACHE_DIR=/workspace/datasets
RUN mkdir -p /workspace/datasets

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Keep the container alive for interactive use
CMD ["tail", "-f", "/dev/null"]