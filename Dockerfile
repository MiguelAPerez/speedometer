FROM python:3.11-slim

# System libs required by OpenCV (headless still needs libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — swap in headless OpenCV to avoid X11 bloat
COPY requirements.txt .
RUN pip install --no-cache-dir \
        $(grep -v "^opencv-python" requirements.txt) \
        opencv-python-headless>=4.9.0

# Copy application source
COPY speedcam/     speedcam/
COPY templates/    templates/
COPY static/       static/
COPY app.py        .

# Persist calibration and uploaded videos across container restarts
VOLUME ["/root/.speedcam"]

# Tell the app where to write user data
ENV SPEEDCAM_DATA_DIR=/root/.speedcam

EXPOSE 8080

# Ensure the data dir and tmp subdir exist at startup
CMD ["sh", "-c", \
     "mkdir -p /root/.speedcam/tmp && \
      python app.py --host 0.0.0.0 --port 8080"]
