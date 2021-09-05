FROM tensorflow/tensorflow:latest-gpu
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation

COPY gretel/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY gretel/install_gretel.py /install_gretel.py
RUN python /install_gretel.py

WORKDIR /dp-tools-evaluation/gretel

COPY data /dp-tools-evaluation/data

COPY dpevaluation/ /usr/local/lib/python3.6/dist-packages/dpevaluation
COPY gretel/evaluation evaluation
COPY gretel/src src

CMD [ "./evaluation/entrypoint.sh" ]