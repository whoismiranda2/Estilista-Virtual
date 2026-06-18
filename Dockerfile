FROM python:3.11-slim

# Libs del sistema: g++ para compilar C++, libgl/libglib para OpenCV/mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Compilar C++ (capa cacheada: solo se recompila si vision/ cambia) ──
COPY vision/ vision/
RUN g++ -O2 -std=c++17 -o procesar \
    vision/main.cpp vision/lector_bmp.cpp vision/histograma.cpp \
    vision/color_dominante.cpp vision/escala_grises.cpp vision/rgb_hsv.cpp \
    vision/features.cpp vision/segmentacion.cpp vision/perceptron.cpp \
    vision/cuerpo.cpp vision/colorimetria.cpp vision/detector_prendas.cpp

# ── 2. Dependencias Python ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Código de la aplicación ──
COPY . .

# Directorio de prendas (Railway Volume se monta aquí)
RUN mkdir -p static/prendas

EXPOSE 8080

# Railway inyecta $PORT en tiempo de ejecución
CMD gunicorn --workers 2 --bind 0.0.0.0:${PORT:-8080} app:app
