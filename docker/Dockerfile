FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

# Fix for current nvidia key issue
# Please copy cuda-keyring file to the docker directory before building docker container.
# For more details on how to download the keys, refer below:
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
COPY docker/cuda-keyring_1.0-1_all.deb /tmp/cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list \
	&& rm /etc/apt/sources.list.d/nvidia-ml.list \
	&& dpkg -i /tmp/cuda-keyring_1.0-1_all.deb

ARG DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
     apt-get install -y wget vim git zip && \ 
     apt-get install -y sudo software-properties-common systemd-sysv 

# Optional but might be needed
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libopenmpi-dev -y 

# Installing Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# Install pytorch libraries
RUN conda create -n py310_pytorch112 python=3.10
SHELL ["/bin/bash", "-c"]
RUN echo $(conda config --get channels)
RUN source activate py310_pytorch112 && \
		echo $(python -V) && \
		conda install -y pytorch==1.12.0 torchvision==0.13.0 -c pytorch 

# DDA and MMCV related libraries
RUN source activate py310_pytorch112 && \
		pip install openmim blobfile tqdm pandas mpi4py cupy-cuda11x mmcls && \
		mim install mmcv-full
		
# Remaining libraries
COPY docker/requirements.txt /tmp/
RUN source activate py310_pytorch112 && \
		pip install -r /tmp/requirements.txt

# Additional installation if required (mmcls)
# git clone https://github.com/open-mmlab/mmclassification.git
# cd mmclassification
# pip3 install -e .