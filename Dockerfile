# https://github.com/python-poetry/poetry/discussions/1879
# to improve ^^

## STAGE 1 - Core package(s)

FROM ghcr.io/pyo3/maturin:main as maturin

RUN mkdir -p /app/build/bonn
WORKDIR /app/build/test_data
# RUN curl -L -O "...wiki/wiki.en.fifu"
WORKDIR /app/build

RUN yum install -y lapack-devel atlas-devel openblas-devel

COPY Cargo.lock /app/build
COPY Cargo.toml /app/build
COPY LICENSE.md /app/build
COPY README.md /app/build

RUN RUSTFLAGS="-L /usr/lib64/atlas -C link-args=-lsatlas -ltatlas -llapack" cargo install finalfusion-utils --features=opq

COPY pyproject.toml /app/build
COPY src /app/build/src
COPY bonn /app/build/bonn

WORKDIR /app/build
