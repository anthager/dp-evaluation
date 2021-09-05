FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

COPY smartnoise-synthetic/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data /dp-tools-evaluation/data
COPY dpevaluation/ /opt/conda/lib/python3.8/site-packages/dpevaluation

WORKDIR /dp-tools-evaluation/smartnoise-synthetic

COPY smartnoise-synthetic/evaluation evaluation
COPY smartnoise-synthetic/src src



CMD [ "./evaluation/entrypoint.sh" ]

