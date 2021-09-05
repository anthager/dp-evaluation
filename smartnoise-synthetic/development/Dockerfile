FROM python:3.6
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

RUN mkdir -p /results

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt install -y make git vim

COPY development/entrypoint.sh development/entrypoint.sh


CMD [ "development/entrypoint.sh" ]