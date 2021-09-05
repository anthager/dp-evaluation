FROM python:3.7
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

RUN mkdir -p /results

COPY google-dp/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data /dp-tools-evaluation/data
COPY dpevaluation/ /usr/local/lib/python3.7/site-packages/dpevaluation

WORKDIR /dp-tools-evaluation/google-dp

COPY google-dp/evaluation evaluation
COPY google-dp/src src



CMD [ "./evaluation/entrypoint.sh" ]

