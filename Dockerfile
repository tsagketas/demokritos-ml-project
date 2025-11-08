FROM python:3.11-slim

WORKDIR /workspace

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Copy requirements file
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]

