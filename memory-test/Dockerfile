FROM python:3.7
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

# install docker in the container. We dont care about the daemon but we do want
# the cli and this was a very easy way of installing it
RUN curl -fsSL https://get.docker.com | bash

COPY memory-test/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY dpevaluation/ /usr/local/lib/python3.7/site-packages/dpevaluation

WORKDIR /dp-tools-evaluation/memory-test

RUN mkdir src
COPY ./**/*_memory_test.sh ./src/
COPY memory-test/src ./src

