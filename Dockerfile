FROM pytorch/pytorch:nightly-devel-cuda9.2-cudnn7


RUN apt-get -qq update && apt-get -qqy upgrade
RUN apt-get -qqy install xserver-xorg-core
RUN apt-get -y install xserver-xorg-video-dummy
RUN apt-get -y install python3-pip
RUN apt-get -y install python3
RUN apt-get -y install x11vnc
RUN apt-get -y install unzip
RUN apt-get -y install pciutils
RUN apt-get -y install software-properties-common
RUN apt-get -y install kmod
RUN apt-get -y install gcc
RUN apt-get -y install make
RUN apt-get -y install linux-headers-generic
RUN apt-get -y install wget
RUN apt-get -y install sudo
RUN apt-get -y install nano


RUN pip install --upgrade \
    pip \
    setuptools



RUN pip install \
    ipykernel \
    jupyter \
    keras \
    matplotlib \
    mock \
    sklearn \
    pandas \
    enum34 \
    cython

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY xorg.conf /etc/X11/xorg.conf


COPY dummy-1920x1080.conf /etc/X11/dummy-1920x1080.conf


COPY . /app
WORKDIR /app


CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
