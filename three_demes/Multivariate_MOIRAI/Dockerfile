
# Use the Python base image (multi-arch)
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the matplotlib backend to 'Agg' for non-GUI backend
ENV MPLBACKEND=Agg

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install JAX and jaxlib compatible with ARM64
RUN pip install --no-cache-dir "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install PyTorch and torchvision
RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install scikit-learn

# Install general Python packages
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    gluonts \
    huggingface_hub

# Install GitHub repositories
COPY uni2ts /app/MOIRAI
RUN pip install --no-cache-dir /app/MOIRAI

# Copy your Python script and data into the container
COPY script.py /app/script.py

# Run the Python script when the container starts
CMD ["python", "script.py"]
