FROM python:3.7
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

RUN mkdir -p /results

RUN apt install -y make git vim

RUN apt install -y python3-pip

COPY development/entrypoint.sh /entrypoint.sh

COPY requirements.txt requirements.txt
RUN python3.7 -m pip install -r requirements.txt

CMD [ "/entrypoint.sh" ]