FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y tzdata && \
    apt-get install -y software-properties-common

RUN apt-get update && \
    apt-get install -y --allow-unauthenticated libsasl2-dev python-dev libldap2-dev libssl-dev openssl ffmpeg \
        build-essential libasound2 libffi-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH /opt/conda/bin:$PATH

# conda installation
RUN apt-get update && apt-get install -y wget

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -n base -c defaults conda

  
RUN apt-get update && apt-get install -y pandoc 
RUN apt-get update && apt-get install -y texlive-xetex
RUN apt-get update && apt-get install -y texlive-fonts-recommended
#RUN apt-get upd ate && apt-get install -y texlive-generic-recommended
#RUN apt-get update && apt-get install -y texlive-generic-extra
RUN apt-get update && apt-get install -y vim
RUN apt-get install build-essential
# Create folders
RUN mkdir /data
RUN mkdir /data/models

# install python packages
RUN pip install streamlit
RUN pip install pandas
RUN pip install numpy
Run pip install torch
Run pip install torchvision
Run pip install matplotlib
Run pip install opencv-python
Run pip install Cython
Run pip install pycocotools
Run pip install Pillow

# copy files inside the docker
COPY entrypoint.sh /data/entrypoint.sh
COPY load_mvp.py /data/load_mvp.py
COPY app.py /data/app.py
COPY engine.py /data/engine.py
COPY utils.py /data/utils.py
COPY transforms.py /data/transforms.py
COPY coco_utils.py /data/coco_utils.py
COPY coco_eval.py /data/coco_eval.py



COPY models/model_mask-trained_cpu.pkl /data/models/model_mask-trained_cpu.pkl



#build volumes
WORKDIR /data


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

EXPOSE 8503


#run entrypoint
CMD sh entrypoint.sh

