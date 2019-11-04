pyastrovis
===============================

WebRTC Fits Streaming

Installation
------------

To install use pip:

    $ pip install pyastrovis
    $ jupyter nbextension enable --py --sys-prefix pyastrovis


For a development installation (requires npm):

    $ git clone https://github.com/ICRAR/pyastrovis.git
    $ cd pyastrovis
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --sys-prefix pyastrovis
    $ jupyter nbextension enable --py --sys-prefix pyastrovis

Docker:

    $ git clone https://github.com/ICRAR/pyastrovis.git
    $ cd pyastrovis
    $ docker build .
    $ docker tag <id> pyastrovis
    $ docker run -it -p 8888:8888 -p 8080:8080 -v <local>:/images pyastrovis