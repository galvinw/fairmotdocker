# Dockerfile.fair

FROM python:3.8.12

WORKDIR /lauretta

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get upgrade -y
# RUN apt-get install -y python3-opencv
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3.8 -m pip install --upgrade pip
RUN pip3.8 install numpy==1.18.4 cython==0.29.24 wheel 
RUN pip3.8 install lap==0.4.0 cython_bbox==0.1.3
RUN pip3.8 install openpifpaf==0.12.14

# install dependencies
COPY requirements.txt .
RUN pip3.8 install -r requirements.txt

COPY . .
RUN python3.8 DCNv2/setup.py install develop

ENV PYTHONPATH "${PYTHONPATH}:src/lib:DCNv2/src:DCNv2"
