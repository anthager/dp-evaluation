# SmartNoise Evaluation

## How to run

### Init docker containers
1. From `smartnoise/`: start docker container with PostgreSQL
```
$ docker-compose up -d
```

2. Build docker image for smartnoise project:
```
$ ./development/build.sh
```

3. Execute [`start.sh`](development/start.sh) script:
```
$ ./development/start.sh
```

### Run smartnoise evaluation
1. Install [`requirements`](requirements.txt)
```
$ python3.7 -m pip install -r requirements.txt
```

2. Execute [`run.sh`](analysis/run.sh) script:
```
$ ./run.sh
```
