# Lauretta Person Reidentification and tracking system - Dockerized


### Requirements

Docker engine

### Setting Up GPU (skip this step if using CPU)
Ubuntu is required for docker to connect with CUDA

Additional steps for Windows:
- Install [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install)

Using Ubuntu bash terminal, install `nvidia-docker2` packages. 
```sh
# The steps below are taken from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Install the nvidia-docker2 package (and dependencies) after updating the package listing:
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to complete the installation after setting the default runtime:
$ sudo systemctl restart docker

# At this point, a working setup can be tested by running a base CUDA container:
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# This should result in a console output shown below which indicates that GPU setup is completed:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Getting Started

Build the images and spin up the containers:
You only need to build it once

1. Download  [monofair_models.zip](https://drive.google.com/file/d/1HDb36ReZc2knfu2nENHp8Fc9Mcq8zuwy/view?usp=sharing), unzip and place all the contents to the `/monofair2/models` folder

2. Edit the `/config/camera.json` file to use your own camera IP address. If you do not have a camera IP address, you may download a demo file from here [API Demo Video](https://www.dropbox.com/s/0c4szm1q9x2a83m/fastapidemoclip.mp4?dl=0)
    ```json
    # Example of camera.json, there must be at least 1 camera entry

    {
        "camera_list": [
            {
                "camera_name": "demo_video",
                "connection_string": "/videos/fastapidemoclip.mp4",
                "position_x": 10.264,
                "position_y": 12.353,
                "position_z": 0.231,
                "focal_length": 2.8
            },
            {
                "camera_name": "demo_rtsp",
                "connection_string": "rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101",
                "position_x": 0.323,
                "position_y": 9.854,
                "position_z": 10.32,
                "focal_length": 4.2
            },
            {
                "camera_name": "demo_camera",
                "connection_string": "http://128.106.126.8:88/webcapture.jpg?command=snap&amp;channel=1?1646104969",
                "position_x": 7.32,
                "position_y": 30.213,
                "position_z": 9.123,
                "focal_length": 3.4
            }
        ]
    }
    ```
3. If you are using the video file, place it in the `/videos/` folder
4. Add any required zones in `zone.json` under `/config/` folder. 
    - The `"camera_name"` has to match with `camera.json` file.
    - Currently, it does not support overlapping zones.
5. Run the following code at parent folder

```sh
$ docker-compose build
$ docker-compose up
```

### Setting up display (imshow)
To view analyzed videos in GUI, run the following command at host machine (any directory)
```sh
xhost +local:
```
Ensure that the following code is uncommented
```sh
# docker-compose.yml

monofair2:
...

    volumes:
        - /tmp/.X11-unix:/tmp/.X11-unix
```

### Testing the APIs

It will run on 
http://localhost:8000

Read the API documentation

Swagger: http://localhost:8000/docs
Redoc: http://localhost:8000/redoc


