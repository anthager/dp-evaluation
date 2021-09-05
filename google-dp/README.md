# Google-DP Evaluation

## How to run

### Init docker containers
1. From `google-dp/`: build custom postgreSQL image (this may take some time) and start docker container
```
$ ./postgres/build.sh
$ docker-compose up -d
```

2. Build docker image for google-dp project:
```
$ ./development/build.sh
```

3. Execute [`start.sh`](development/start.sh) script:
```
$ ./development/start.sh
```

### Run google-dp evaluation
1. Install [`requirements`](requirements.txt)
```
$ python3.7 -m pip install -r requirements.txt
```

2. TODO
```
$ ...
```
