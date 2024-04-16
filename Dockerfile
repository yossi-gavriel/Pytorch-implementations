# Using an official PyTorch image with CUDA support
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run on container start
CMD ["python", "face.py"]
