FROM python:3.7
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

COPY diffprivlib/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data /dp-tools-evaluation/data
COPY dpevaluation/ /usr/local/lib/python3.7/site-packages/dpevaluation

WORKDIR /dp-tools-evaluation/diffprivlib

COPY diffprivlib/evaluation evaluation
COPY diffprivlib/src src

CMD [ "./evaluation/entrypoint.sh" ] 