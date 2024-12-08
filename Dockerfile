# Use the official Python image with a slim OS
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libcairo2-dev \
    libpango1.0-0 \
    libpangocairo-1.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Manim Community Edition
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir manim

# Set the working directory
WORKDIR /manim

# Copy your project files into the container
COPY . /manim

# Set the default command to run when the container starts
CMD ["manim", "-pql", "kyber_visualization.py", "ShowKyberProcess"]
