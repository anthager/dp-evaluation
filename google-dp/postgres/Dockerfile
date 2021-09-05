FROM python:3.7
RUN apt-get update
LABEL org.opencontainers.image.source https://github.com/anthager/dp-tools-evaluation

# install fun stuff
RUN apt update
RUN apt install -y curl gnupg make git openjdk-11-jdk gcc

# install bazel
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
RUN mv bazel.gpg /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt update 
RUN apt install -y bazel-3.7.2
RUN ln -s /usr/bin/bazel-3.7.2 /usr/bin/bazel

# parser and lexer for postgres, why these arent shipped with the installation is beyond my understanding, maybe they are but not in path
RUN apt install -y bison flex postgresql

# set up the repo
RUN git clone https://github.com/google/differential-privacy.git
WORKDIR /differential-privacy/cc

# the install script runs with sudo and the container dont have a sudo command 
# so we make an no op scrip that just runs all args
RUN echo '#!/bin/bash\neval $@' > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo

# copy over needed files
COPY entrypoint.sh /entrypoint.sh
COPY pg-password /pg-password

ENV PATH "/usr/lib/postgresql/11/bin:${PATH}"
ENV PGDATA "/usr/lib/postgresql/11/data"

# set up the database
RUN mkdir ${PGDATA}
RUN chown postgres ${PGDATA}
USER postgres
RUN initdb --pwfile /pg-password --pgdata ${PGDATA}

USER root
# Adjust PostgreSQL configuration so that remote connections to the
# database are possible.
RUN echo "host all  all    0.0.0.0/0  md5" >> ${PGDATA}/pg_hba.conf
# And add ``listen_addresses`` to ``/usr/lib/postgresql/11/data/postgresql.conf``
RUN echo "listen_addresses='*'" >> ${PGDATA}/postgresql.conf

# install the extension
RUN ./postgres/install_extension.sh

USER postgres
CMD [ "/entrypoint.sh" ]

