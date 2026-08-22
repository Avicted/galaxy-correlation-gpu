# syntax=docker/dockerfile:1

# Development image for the galaxy correlation solver.
#
# Provides the CUDA toolchain plus the lint tooling the pre-commit hooks expect
# (clang-format, clang-tidy, shellcheck, hadolint, gitleaks, pre-commit), so
# that a contributor does not have to install any of it on the host.
#
# Build:  make docker
# Lint:   docker run --rm -v "$PWD:/work" galaxy-correlation:dev make lint
# Run:    docker run --rm --gpus all -v "$PWD:/work" galaxy-correlation:dev make verify
#
# Building and linting work without a GPU. Only the run/bench/verify targets
# need --gpus all and a matching host driver.
#
# The CUDA version here matches the toolchain the published figures were
# measured with; see the header of src/galaxy_cuda.cu.

FROM docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG HADOLINT_VERSION=2.12.0
# Pinned exactly: clang-format output changes between major versions, so an
# unpinned version would silently reformat the tree and fight the host.
ARG CLANG_FORMAT_VERSION=22.1.8
ARG GITLEAKS_VERSION=8.24.0

# hadolint ignore=DL3008
RUN <<EOF
    set -euo pipefail
    apt-get update
    apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        clang-tidy \
        curl \
        git \
        make \
        python3 \
        python3-pip \
        python3-venv \
        shellcheck
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# hadolint, as a static binary rather than an apt package.
RUN <<EOF
    set -euo pipefail
    curl -fsSL -o /usr/local/bin/hadolint \
        "https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-Linux-x86_64"
    chmod +x /usr/local/bin/hadolint
    hadolint --version
EOF

# gitleaks, for the secret-scanning pre-commit hook.
RUN <<EOF
    set -euo pipefail
    curl -fsSL -o /tmp/gitleaks.tar.gz \
        "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
    tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks
    rm /tmp/gitleaks.tar.gz
    chmod +x /usr/local/bin/gitleaks
    gitleaks version
EOF

# pre-commit itself. Ubuntu 24.04 marks the system Python as externally managed,
# so install into a venv rather than fighting PEP 668.
ENV VIRTUAL_ENV=/opt/venv
RUN <<EOF
    set -euo pipefail
    python3 -m venv "$VIRTUAL_ENV"
EOF

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# hadolint ignore=DL3013
RUN <<EOF
    set -euo pipefail
    pip install --no-cache-dir pre-commit "clang-format==${CLANG_FORMAT_VERSION}"
    clang-format --version
EOF

# Run as a non-root user so files created in the bind mount stay owned by the
# caller on a typical single-user host.
ARG UID=1001
ARG GID=1001
RUN <<EOF
set -euo pipefail
groupadd --gid "$GID" dev
useradd --uid "$UID" --gid "$GID" --create-home --shell /bin/bash dev
EOF

USER dev

WORKDIR /work

CMD ["make", "help"]
