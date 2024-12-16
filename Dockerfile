FROM mambaorg/micromamba:latest
USER root
RUN apt-get update && apt-get install -y gcc && apt-get install -y g++ 
RUN apt install -y libssl-dev 
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install pyarmor==6.7.4 scikit-learn==1.3.* pytorch=2.0.1
WORKDIR /code
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform