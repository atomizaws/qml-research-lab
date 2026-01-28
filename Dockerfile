# Use Python 3.10 slim base image (ARM64 compatible)
FROM python:3.10-slim

# Environment variables for optimization and cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# System dependencies
# Install build-essential and libomp-dev for building quantum simulators (Qiskit/PennyLane)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 3. [Critical Step for Parallelization]
# Uninstall pennylane-lightning and force reinstall from source.
# This ensures it links correctly against the system's libomp-dev for multi-core support.
RUN pip uninstall -y pennylane-lightning && \
    pip install --no-binary pennylane-lightning pennylane-lightning

# User configuration
RUN useradd -m researcher
USER researcher

# JupyterLab configuration
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--no-browser"]
