FROM tensorflow/tensorflow:latest-gpu
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

COPY tensorflow-privacy/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data /dp-tools-evaluation/data
COPY dpevaluation/ /usr/local/lib/python3.6/dist-packages/dpevaluation

WORKDIR /dp-tools-evaluation/tensorflow-privacy

COPY tensorflow-privacy/evaluation evaluation
COPY tensorflow-privacy/src src



CMD [ "./evaluation/entrypoint.sh" ]

