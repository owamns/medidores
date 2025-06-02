# Usar una imagen base de Python con OpenCV preinstalado
FROM python:3.12-slim

# Instalar dependencias del sistema necesarias para OpenCV y otras librerías
RUN apt-get update && apt-get install -y \
  libopencv-dev \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgcc-s1 \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para los modelos
RUN mkdir -p models

# Copiar el código de la aplicación
COPY app.py .
COPY process_images.py .
COPY gunicorn.conf.py .

# Copiar los modelos (asegúrate de que existan en tu proyecto)
COPY models/ ./models/

# Crear un usuario no-root para seguridad
RUN useradd -m -u 1000 appuser

# Crear directorio de configuración para Ultralytics y darle permisos
RUN mkdir -p /home/appuser/.config/Ultralytics && \
  chown -R appuser:appuser /home/appuser/.config

# Cambiar permisos de la aplicación
RUN chown -R appuser:appuser /app

USER appuser

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
