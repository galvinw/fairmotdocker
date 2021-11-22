# Dockerfile.web

FROM python:3.8.12

WORKDIR /lauretta

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get upgrade -y

COPY . .

# install dependencies
RUN python3.8 -m pip install --upgrade pip
RUN pip3.8 install -r requirements.txt
RUN apt-get update && apt-get install -y netcat

