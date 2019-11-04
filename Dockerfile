FROM ubuntu:18.04

RUN apt-get update && apt-get -y upgrade && apt-get install -y ffmpeg git python3.7 python3-venv
RUN apt-get install -y python3-pip curl
RUN apt-get install -y sudo pkg-config libavdevice-dev libopus-dev opus-tools libogg-dev libvpx-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
RUN update-alternatives --config python3
RUN python3 --version
RUN apt-get install -y python3.7-dev
RUN pip3 install --upgrade pip
RUN curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
RUN apt-get install -y nodejs
RUN rm -rf /pyastrovis/
RUN git clone https://github.com/ICRAR/pyastrovis.git /pyastrovis/
WORKDIR /pyastrovis/
RUN pip3 install -e .
RUN jupyter nbextension install --py --symlink --sys-prefix pyastrovis
RUN jupyter nbextension enable --py --sys-prefix pyastrovis
RUN mkdir -p /images
RUN jupyter trust example.ipynb
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]