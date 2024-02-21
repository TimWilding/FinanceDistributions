FROM continuumio/miniconda3:latest AS miniconda
WORKDIR /fastdistribution
COPY . .
RUN conda env create -n fastdistribution --file environment.yml \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
	&& activate fastdistribution
RUN pip install .

