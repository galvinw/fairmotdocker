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
RUN python3 setup.py bdist_wheel
RUN pip3 install -e .

RUN python3.8 visuals/FairMOT/DCNv2/setup.py install develop

ENV PYTHONPATH "${PYTHONPATH}:visuals/FairMOT/src/lib:visuals/FairMOT/DCNv2/src:visuals/FairMOT/DCNv2"

CMD ["python3", "-m", "monoloco.run", "predict", "--webcam", "/videos/fastapidemoclip.mp4", "--activities", "social_distance", "--output_types", "multi"]
