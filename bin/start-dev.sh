#!/usr/bin/env bash
set -e

GREEN='\e[32m'
DEFAULT='\e[0m'
STRING='[INFO]'
INFO=$GREEN$STRING$DEFAULT
TEXT="Running docker container \"$CONTAINER_NAME\" in background on"

# 'dev' is not running -> start it in detached mode
# This makes sure it will keep running even when the person starting container exits it
if [[ -z $(docker ps --format '{{.Names}}' | grep $CONTAINER_NAME) ]]; then
  # For Mac OS we need to use this magic path, I have no idea why
  if [[ "$OSTYPE" == "darwin"* ]]; then
    SOCKET_PATH=/run/host-services/ssh-auth.sock
    SOCKET_DIR=/run/host-services/ssh-auth.sock
    echo -e "$INFO $TEXT Mac..."
  else
    SOCKET_PATH=$SSH_AUTH_SOCK
    SOCKET_DIR=$(dirname ${SSH_AUTH_SOCK} 2>/dev/null &)
    echo -e "$INFO $TEXT Linux..."
  fi
  # Pull image if no arg is provided
  if [ -z "$1" ]; then
    docker pull $IMAGE
  fi
  docker run \
    -itd \
    --name $CONTAINER_NAME \
    --network $NETWORK \
    -v $SOCKET_DIR:$SOCKET_DIR \
    -v results:/results \
    -v data:/data \
    -e SSH_AUTH_SOCK=$SOCKET_PATH \
    $IMAGE
fi

echo -e "$INFO Starting docker container \"$CONTAINER_NAME\"..."

docker exec \
  -it \
  $CONTAINER_NAME \
  /bin/bash
