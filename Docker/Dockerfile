# FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

ENV CUDACXX=/usr/local/cuda-11.2/bin/nvcc
RUN apt-get update
RUN apt-get install -y  wget \
                        libssl-dev \
                        libjpeg-dev 
RUN rm -rf /var/lib/apt/lists/*

ENV version=3.20
ENV build=0
RUN mkdir /cmake_install
WORKDIR /cmake_install
RUN wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
RUN tar -xzvf cmake-$version.$build.tar.gz

WORKDIR /cmake_install/cmake-$version.$build/
RUN ./bootstrap
RUN make -j$(nproc) && make install

RUN apt-get update && apt-get install -y python3.6
RUN ln -s /usr/bin/python3.6 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /apnn-tc