# Lauretta Person Reidentification and tracking system - Dockerized


### Requirements

Docker engine

### Getting Started

Build the images and spin up the containers:
You only need to build it once

1. Copy  [DLA-34 official](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view) to the `monofair/modelsx` folder
2. Edit the `config/camera.txt` file to use your own camera IP address. If you do not have a camera IP address, you may download a demo file from here [API Demo Video](https://www.dropbox.com/s/0c4szm1q9x2a83m/fastapidemoclip.mp4?dl=0)
3. If you are using the video file, place it in the `videos/` folder
4. Run the following code at parent folder

```sh
$ docker-compose build
$ docker-compose up
```

### Testing the APIs

It will run on 
http://localhost:8000

Read the API documentation

Swagger: http://localhost:8000/docs
Redoc: http://localhost:8000/redoc


