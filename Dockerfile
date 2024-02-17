FROM continuumio/miniconda3:latest AS miniconda
WORKDIR /fastdistribution
COPY . .
RUN conda config --append channels conda-forge \
    && conda install --file requirements.txt \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete
RUN pip install .
