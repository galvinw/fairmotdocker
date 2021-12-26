# Dockerfile.monofair2
FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils python3.8-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

# FROM python:3.8.12

WORKDIR /lauretta

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3.8 -m pip install --upgrade pip
RUN pip3.8 install numpy==1.18.4 cython==0.29.24 wheel 
RUN pip3.8 install lap==0.4.0 cython_bbox==0.1.3 wheel
RUN pip3.8 install openpifpaf==0.12.14
# RUN pip3.8 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3.8 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY requirements.txt .
RUN pip3.8 install -r requirements.txt

COPY . .

# run setup scripts
RUN (cd monoloco ; python3.8 setup.py bdist_wheel)
RUN (cd monoloco ; pip3.8 install -e .)

ENV PYTHONPATH "${PYTHONPATH}:$PWD/fairmot/src/lib:$PWD/fairmot/DCNv2/src:$PWD/fairmot/DCNv2:build/bdist.linux-x86_64/egg"
RUN (cd fairmot ; python3.8 DCNv2/setup.py install develop --user)

