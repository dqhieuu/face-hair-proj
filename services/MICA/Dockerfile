FROM ubuntu:22.04
COPY . .
RUN rm -rf external
COPY external/.insightface ~/
RUN apt update && apt install -y wget build-essential

# Install conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN mamba env create -f environment.yml
# Activate conda environment
SHELL ["conda", "run", "-n", "MICA", "/bin/bash", "-c"]
RUN conda activate MICA

ENTRYPOINT [ "python", "demo.py" ]

