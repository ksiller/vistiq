FROM pytorch/pytorch:latest
LABEL org.opencontainers.image.source="https://github.com/uvarc/vistiq"

WORKDIR /opt/vistiq

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gpg \
    git=1:2.* \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install UV from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /bin/uv

# Copy the repository
COPY . ./

# Need to build sdist to dynamically set version with versioningit
RUN rm -rf dist && \
    uv build --sdist --out-dir dist
RUN mv "dist/vistiq-"*".tar.gz" "dist/vistiq.tar.gz"

# install from dist
RUN uv pip install --system "./dist/vistiq.tar.gz" && \
    rm -rf dist/
    
# Smoke test
RUN vistiq -h

# Setup entrypoint
COPY scripts/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/vistiq/entrypoint.sh"]


