# ==========================================
# Stage 1: Build Stage (Frontend Bundler & Prolog/Python)
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

RUN git clone https://github.com/trealla-prolog/trealla.git /tmp/trealla \
    && cd /tmp/trealla \
    && make \
    && make install

COPY frontend/ /build/frontend/
WORKDIR /build/frontend
RUN npm install && npm run build

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
    glfw


# ==========================================
# ==========================================
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libedit2 \
    libffi8 \
    libssl3 \
    ffmpeg \
    libgl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/tpl /usr/local/bin/tpl
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /build/frontend/dist /app/frontend/dist
COPY . /app

EXPOSE 8000

CMD ["python", "model.py"]