# hash:sha256:5a5b52110e8f6ba05d057f8341cd589a477f82af571e7b4b13c00b455429ac3f
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages for LaTeX rendering, OpenCV, and development headers
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        cm-super \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy project
COPY . /workspace

ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV HDF5_USE_FILE_LOCKING=FALSE

CMD ["/bin/bash"]
