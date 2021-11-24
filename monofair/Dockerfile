# Dockerfile.fair

FROM python:3.8.12

WORKDIR /lauretta

# Install dependencies
RUN pip3.8 install wheel
RUN pip3.8 install sdist

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-opencv
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip3.8 install -r requirements.txt
RUN pip3.8 install cython_bbox

COPY . .
RUN python3 setup.py sdist bdist_wheel
RUN (cd monoloco/visuals/FairMOT/DCNv2-pytorch_1.6/ ; chmod +x make.sh ; ./make.sh)