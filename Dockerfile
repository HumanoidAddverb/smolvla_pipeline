# Author : Rajesh Kumar (rajesh.kumar01@addverb.com)
# Description : Dockerfile supplied to IIITH for swift deployment of RL Models 

# Initial configurations 
FROM ubuntu:focal as build

# Timezone is Indian 
ENV TZ=Indian

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install OpenGL dependencies

RUN apt-get update && apt-get install -y libgl1-mesa-dev

# Install python 3 
RUN apt-get install -y python3
RUN apt-get install -y python3-pip




# Install dependencies
RUN apt-get install -y libglfw3 libglfw3-dev


RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    libeigen3-dev 

# Setup pip stuff 
RUN pip install mujoco==2.3.7
RUN pip install stable-baselines3[extra]
RUN pip install imageio

# Setting the directories 
COPY . /app

WORKDIR /app/training_script

RUN apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    curl \
    xvfb \
    ffmpeg \
    xorg-dev \
    libsdl2-dev \
    swig \
    cmake \
    python-opengl

RUN apt-get install -y x11-apps

CMD [ "/bin/bash" ]