FROM python:3.6
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation
RUN apt-get update

COPY synthetic-evaluator/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data /data
COPY dpevaluation/ /usr/local/lib/python3.6/site-packages/dpevaluation

WORKDIR /src
COPY synthetic-evaluator/entrypoint.sh entrypoint.sh
COPY synthetic-evaluator/src .

CMD [ "./entrypoint.sh" ]