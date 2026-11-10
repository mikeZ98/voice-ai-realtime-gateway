# syntax=docker/dockerfile:1.27@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
#
# Voice gateway image: multi-stage, non-root, digest-pinned, bit-for-bit reproducible.
#
#   docker build -f deploy/Dockerfile --build-arg SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" .
#
# Every input is pinned by content: the frontend, base and uv images by sha256 digest,
# the ASGI runtime by hash (requirements-runtime.txt, --require-hashes). Every timestamp
# is clamped to SOURCE_DATE_EPOCH. Two cold builds of one commit export identical layers
# and an identical manifest when built with BuildKit's rewrite-timestamp exporter option;
# scripts/verify_reproducible_image.sh runs exactly that twice and compares the bytes.
#
# * build stage: uv builds the wheel; a venv receives the hash-locked ASGI server and the
#   gateway wheel (--no-deps: it has zero dependencies). uv's uv_cache.json (a wall-clock
#   stamp for local-path installs) is purged, bytecode is compiled with checked-hash
#   invalidation (a source hash, not an mtime), and the runtime user is written straight
#   into passwd/group (useradd would date-stamp /etc/shadow).
# * runtime stage: only the venv and those two files; no compiler, no uv, no caches, no
#   build context. Runs as uid/gid 10001 on a read-only root filesystem (nothing is
#   written at runtime; /tmp may be a tmpfs).
#
# Shutdown: the Kubernetes preStop hook runs `python -m voice_ai_realtime_gateway.deploy.drain`,
# which sends SIGUSR1 (graceful drain) and waits for /drainz before SIGTERM arrives.
# The offline CLI is also inside: `python -m voice_ai_realtime_gateway --fast --no-hud`.

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- build
FROM ${PYTHON_IMAGE} AS build
ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0
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
 && for cache in /opt/gateway/lib/python3.12/site-packages/*.dist-info/uv_cache.json; do \
        [ -e "$cache" ] || continue; \
        sed -i '/\/uv_cache\.json,/d' "$(dirname "$cache")/RECORD"; \
        rm "$cache"; \
    done \
 && [ -z "$(find /opt/gateway -name uv_cache.json)" ] \
 && /opt/gateway/bin/python -m compileall -q -j 1 --invalidation-mode checked-hash /opt/gateway/lib \
 && /opt/gateway/bin/python -c "import voice_ai_realtime_gateway.deploy.app, uvicorn, websockets"
RUN cp /etc/passwd /etc/group /build/ \
 && printf 'gateway:x:10001:10001:voice gateway:/nonexistent:/usr/sbin/nologin\n' >> /build/passwd \
 && printf 'gateway:x:10001:\n' >> /build/group \
 && find /opt/gateway /build/passwd /build/group -newermt "@${SOURCE_DATE_EPOCH}" \
        -exec touch --no-dereference --date="@${SOURCE_DATE_EPOCH}" {} +

# --------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
ARG SOURCE_DATE_EPOCH=0
LABEL org.opencontainers.image.title="voice-ai-realtime-gateway" \
      org.opencontainers.image.description="Full-duplex voice agent gateway" \
      org.opencontainers.image.source="https://github.com/mikeZ98/voice-ai-realtime-gateway"
COPY --from=build /build/passwd /etc/passwd
COPY --from=build /build/group /etc/group
COPY --from=build --chown=root:root /opt/gateway /opt/gateway
ENV PATH=/opt/gateway/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HOME=/tmp
USER 10001:10001
WORKDIR /tmp
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["--factory", "voice_ai_realtime_gateway.deploy.app:create_demo_app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--ws", "websockets", "--ws-max-size", "1048576", \
     "--lifespan", "on", "--timeout-graceful-shutdown", "10", \
     "--no-access-log", "--no-server-header"]
