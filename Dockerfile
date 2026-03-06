# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Render sets PORT env, but good for local testing)
EXPOSE 5000

# Run Gunicorn server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "3"]