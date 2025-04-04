# Use an official Python image as the base
FROM python:3.9

# Install FFmpeg and other OS-level dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Copy .env file to the container
COPY .env .env
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (change if needed)
EXPOSE 8027

# Command to run the API (modify based on framework)
CMD ["python", "Api.py"]