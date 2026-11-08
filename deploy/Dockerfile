# syntax=docker/dockerfile:1.7
#
# Voice gateway image: multi-stage, non-root, hash-pinned, deterministic bytecode.
#
#   docker build -f deploy/Dockerfile -t voice-gateway .
#
# * build stage: uv builds the wheel; a venv receives the hash-locked ASGI server
#   (--require-hashes) and the gateway wheel (--no-deps: it has zero dependencies).
#   Bytecode is compiled with checked-hash invalidation, so .pyc files carry a source
#   hash instead of an mtime and identical sources give byte-identical layers.
# * runtime stage: only the venv is copied; no compiler, no uv, no pip/uv caches,
#   no build context. Runs as uid/gid 10001 and works on a read-only root filesystem
#   (PYTHONDONTWRITEBYTECODE; nothing is written at runtime).
#
# Shutdown: the Kubernetes preStop hook runs `python -m voice_ai_realtime_gateway.deploy.drain`,
# which sends SIGUSR1 (graceful drain) and waits for /drainz before SIGTERM arrives.

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.11.7 AS uv

# --------------------------------------------------------------------------- build
FROM ${PYTHON_IMAGE} AS build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    SOURCE_DATE_EPOCH=0
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY deploy/requirements-runtime.txt ./
RUN uv build --wheel --out-dir /build/dist \
 && python -m venv --without-pip /opt/gateway \
 && uv pip install --python /opt/gateway/bin/python --require-hashes --no-deps \
        -r requirements-runtime.txt \
 && uv pip install --python /opt/gateway/bin/python --no-deps /build/dist/*.whl \
 && /opt/gateway/bin/python -m compileall -q -j 0 --invalidation-mode checked-hash /opt/gateway/lib \
 && /opt/gateway/bin/python -c "import voice_ai_realtime_gateway.deploy.app, uvicorn, websockets"

# --------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="voice-ai-realtime-gateway" \
      org.opencontainers.image.description="Full-duplex voice agent gateway" \
      org.opencontainers.image.source="https://github.com/mikeZ98/voice-ai-realtime-gateway"
ENV PATH=/opt/gateway/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1
RUN groupadd --system --gid 10001 gateway \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin gateway
COPY --from=build --chown=root:root /opt/gateway /opt/gateway
USER 10001:10001
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["--factory", "voice_ai_realtime_gateway.deploy.app:create_demo_app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--ws", "websockets", "--ws-max-size", "1048576", \
     "--lifespan", "on", "--timeout-graceful-shutdown", "10", \
     "--no-access-log", "--no-server-header"]
