FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-venv \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual para evitar conflictos con paquetes del sistema
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar pip y actualizarlo
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

# Instalar PyTorch con soporte CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalar el resto de dependencias en el entorno virtual
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos de la aplicación
COPY app.py .
COPY process_images.py .
COPY gunicorn.conf.py .
COPY models/ ./models/

# Crear directorios necesarios para el usuario
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.config/Ultralytics /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser/.config /home/appuser/.cache /app

# Cambiar al usuario no-root
USER appuser

# Comando por defecto
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]