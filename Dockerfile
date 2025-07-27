FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    gcc \\
    libpq-dev \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra fast dependency resolver and installer)
RUN pip install --upgrade pip && pip install uv

# Copy project files
COPY . .

# Create virtual environment using uv
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies using uv
RUN uv pip install --upgrade pip
RUN uv pip install -r requirements.txt

# Set environment variables required by Airflow if needed
ENV AIRFLOW_HOME=/app/airflow

# Make required directories
RUN mkdir -p $AIRFLOW_HOME/dags $AIRFLOW_HOME/logs $AIRFLOW_HOME/plugins

# Expose relevant ports
EXPOSE 8000 8080 8793 8794

# Default command for FastAPI app
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]