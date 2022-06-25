FROM python:3.10.4-alpine3.15

RUN apk add alpine-sdk \
            cython \
            openblas-dev \
            zlib-dev \
            jpeg-dev \
            libffi-dev

ADD ./requirements.txt /pip/requirements.txt

RUN pip install -r /pip/requirements.txt

CMD ["ash"]
