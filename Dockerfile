# syntax=docker/dockerfile:1
# VCF Analysis Agent — multi-stage Dockerfile.
# Stages: builder (compiles htslib/bcftools/cyvcf2) → runtime → development.
#
# IMPORTANT: the project's runtime deps (kuzu, lancedb, strands, cyvcf2, ...)
# are pinned in requirements.txt, NOT in pyproject.toml [project.dependencies]
# (which only lists pytest/pre-commit). The runtime stage therefore installs
# from requirements.txt explicitly, then the project wheel on top.

ARG PYTHON_VERSION=3.11
ARG BCFTOOLS_VERSION=1.19
ARG HTSLIB_VERSION=1.19

# =================== Stage 1: builder ======================================
FROM python:${PYTHON_VERSION}-slim AS builder

ARG BCFTOOLS_VERSION
ARG HTSLIB_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ make cmake autoconf automake pkg-config \
        git curl ca-certificates \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
        libdeflate-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build
# htslib ships bgzip + tabix binaries (needed for region queries & indexing).
RUN curl -fsSL https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2 -o htslib.tar.bz2 \
    && tar -xjf htslib.tar.bz2 \
    && cd htslib-${HTSLIB_VERSION} \
    && ./configure --prefix=/usr/local --disable-libcurl \
    && make -j"$(nproc)" && make install && ldconfig

RUN curl -fsSL https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2 -o bcftools.tar.bz2 \
    && tar -xjf bcftools.tar.bz2 \
    && cd bcftools-${BCFTOOLS_VERSION} \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" && make install && ldconfig

WORKDIR /build
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
RUN pip install --upgrade pip wheel \
    && HTSLIB_MODE=external pip wheel --wheel-dir=/opt/wheels -r requirements.txt \
    && pip wheel --wheel-dir=/opt/wheels pytest-cov \
    && pip wheel --wheel-dir=/opt/wheels --no-deps ./

# =================== Stage 2: runtime ======================================
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="vcf-analysis-agent" \
      org.opencontainers.image.description="AI-powered VCF analysis agent (GLM-5.2 + bge-m3 embeddings)" \
      org.opencontainers.image.source="https://github.com/imdeniil/vcf_analysis"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANCEDB_PATH=/app/lancedb \
    KUZU_PATH=/app/kuzu_db

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 libssl3 libbz2-1.0 liblzma5 zlib1g libdeflate0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/wheels /tmp/wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links=/tmp/wheels -r /tmp/requirements.txt \
    && pip install --no-index --find-links=/tmp/wheels --no-deps vcf_analysis_agent \
    && rm -rf /tmp/wheels /tmp/requirements.txt

# bcftools + htslib (also installs bgzip + tabix for region queries / indexing).
COPY --from=builder /usr/local/bin/bcftools /usr/local/bin/bcftools
COPY --from=builder /usr/local/bin/bgzip /usr/local/bin/bgzip
COPY --from=builder /usr/local/bin/tabix /usr/local/bin/tabix
COPY --from=builder /usr/local/bin/htsfile /usr/local/bin/htsfile
COPY --from=builder /usr/local/lib/libhts* /usr/local/lib/
RUN ldconfig && bcftools --version | head -1 && tabix --version | head -1 && bgzip --version | head -1

WORKDIR /app
RUN mkdir -p /app/data /app/lancedb /app/kuzu_db /app/sample_data

CMD ["python", "-c", "import vcf_agent; print('VCF Analysis Agent ready')"]

# =================== Stage 3: development ==================================
FROM runtime AS development

RUN pip install --quiet ipython black ruff mypy pre-commit

WORKDIR /app
CMD ["bash"]
