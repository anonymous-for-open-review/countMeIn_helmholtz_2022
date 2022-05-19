#from  nvcr.io/nvidia/pytorch:22.03-py3
# From here: https://github.com/thinkWhere/GDAL-Docker/blob/develop/3.8-ubuntu/Dockerfile
from ubuntu:20.04
# Update base container install
run echo "hi"
RUN apt-get update
#RUN apt-get upgrade -y

ENV TZ 'GB'
RUN echo $TZ > /etc/timezone && \
    apt-get install -y tzdata && \
    rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

# Install GDAL dependencies
RUN apt-get install -y python3-pip libgdal-dev locales ffmpeg libsm6 libxext6 

# Ensure locales configured correctly
RUN locale-gen en_GB.UTF-8
ENV LC_ALL='en_GB.utf8'

# Set python aliases for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install latest version of GDAL
RUN pip3 install numpy
RUN pip3 install GDAL==3.0.4

RUN pip3 install sklearn
RUN pip3 install opencv-python pandas rasterio matplotlib 
