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
RUN pip install --no-cache-dir --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install PyTorch (will automatically select ARM64 version)
RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install numpy 
# Install other dependencies
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    gluonts \
    huggingface_hub \
    git+https://github.com/SalesforceAIResearch/uni2ts.git

# Copy your Python script and data files into the container (this is not necessary when using orchestrator becuase I mount the entire folder)
COPY script.py /app/script.py 


# Run the Python script when the container starts
CMD ["python", "script.py"]
