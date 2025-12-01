FROM rust:slim

# install system deps
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    protobuf-compiler \
    linux-perf && \
    rm -rf /var/lib/apt/lists/*

# set up python venv
RUN python3.13 -m venv /venv
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"

# install maturin and dependencies
RUN pip install --upgrade pip && \
    pip install "maturin>=1.8,<2.0" typer patchelf && \
    pip install turbopuffer==1.5.0 topk-sdk==0.7.0 pymilvus==2.5.17 pinecone[grpc]==7.3.0 qdrant-client==1.15.1

# copy source code
WORKDIR /sdk
COPY . .

# build and install topk_bench using maturin, also mount pip cache and wheels
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/sdk/target,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip/wheels,sharing=locked \
    maturin develop --release
