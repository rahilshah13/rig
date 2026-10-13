# ==========================================
# Stage 1: Build Stage (Frontend Bundler & Trealla Prolog & Python Dependencies)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libedit-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Clone and build Trealla Prolog so 'tpl' is available
RUN git clone https://github.com/trealla-prolog/trealla.git /tmp/trealla \
    && cd /tmp/trealla \
    && make \
    && make install

# Copy frontend source and build production bundle
COPY frontend/ /build/frontend/
WORKDIR /build/frontend
RUN npm install && npm run build

# Install Python packages in builder stage
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "numpy<2" \
    jax \
    flax \
    optax \
    trimesh \
    matplotlib \
    fastapi \
    uvicorn \
    requests \
    PyOpenGL \
    glfw \
    imageio


# ==========================================
# Stage 2: Final Runtime Image
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# Install necessary runtime libraries (FFmpeg for video/GIF encoding, GL/GLFW bindings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libedit2 \
    libffi8 \
    libssl3 \
    ffmpeg \
    libgl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries, Python site-packages, and built frontend from builder stage
COPY --from=builder /usr/local/bin/tpl /usr/local/bin/tpl
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /build/frontend/dist /app/frontend/dist
COPY . /app

# Create persistent checkpoints directory and robust entrypoint script
RUN mkdir -p /app/checkpoints

RUN printf '#!/bin/sh\n\
set -e\n\
mkdir -p /app/checkpoints\n\
if [ ! -f /app/checkpoints/checkpoint_bundle.pickle ]; then\n\
    echo "[Entrypoint] No checkpoint found in volume. Initializing blank placeholder..."\n\
    touch /app/checkpoints/checkpoint_bundle.pickle\n\
fi\n\
exec "$@"\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000
EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sh", "-c", "python render.py 8001 & python model.py"]