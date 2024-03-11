ARG RUST_VERSION="1.75"
FROM rust:${RUST_VERSION}

### Use bash as the default shell
SHELL ["/bin/bash", "-c"]

### Configure the workspace
ARG WORKSPACE="/root/ws"
ENV WORKSPACE="${WORKSPACE}"
WORKDIR ${WORKSPACE}

### Install Python
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

### Install Gymnasium
ARG GYMNASIUM_VERSION="0.29.1"
RUN python3 -m pip install --break-system-packages --no-cache-dir "gymnasium[all]==${GYMNASIUM_VERSION}"

### Copy the source
COPY . "${WORKSPACE}"

### Build the project
RUN cargo build --release --all-targets --all-features
