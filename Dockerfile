FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential \
                       python3.9-dev \
                       python3-pip \
                       cython3 \
                       libopenblas-dev \
                       zlib1g-dev \
                       libjpeg-dev \
                       libffi-dev

ADD ./requirements.txt /pip/requirements.txt

RUN pip install -r /pip/requirements.txt

CMD ["ash"]
