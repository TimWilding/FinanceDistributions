FROM continuumio/miniconda3:latest AS miniconda

WORKDIR /tmp

RUN apt-get update && \
 apt-get install -yq --no-install-recommends curl && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*

COPY . financedistributions
WORKDIR financedistributions


RUN conda env create -n fastdistribution --file environment.yml \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
	&& activate fastdistribution
RUN conda install jupyter
RUN pip install .

ARG TEST
RUN if [[ -n "${TEST}" ]] ; then \
    pip install --no-cache-dir pytest && \
    python -m pytest tests ; fi


EXPOSE 8888
CMD ["/bin/sh",  "start_jupiter.sh"]