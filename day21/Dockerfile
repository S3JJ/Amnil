# Using Python-Slim which is a lightweight Python image
FROM python:3.11-slim

# Setting required environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Setting working directory inside container
WORKDIR /app

# Copying only requirement files first 
COPY requirements.txt .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the project files
COPY . .

# Exposing port 
EXPOSE 8000

# Command to run FastAPI using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
