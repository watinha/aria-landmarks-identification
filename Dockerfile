FROM python:alpine

RUN apk add alpine-sdk \
            cython \
            openblas-dev \
            zlib-dev \
            jpeg-dev \
            libffi-dev
RUN pip install selenium \
                numpy \
                pandas \
                sklearn \
                imblearn \
                openpyxl \
                Pillow

CMD ["ash"]
