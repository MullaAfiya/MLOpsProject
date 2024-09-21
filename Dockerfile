# # Use the official Python image
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy requirements.txt and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Command to run the application
# CMD ["python", "model.py"]

FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:5000/predict || exit 1
