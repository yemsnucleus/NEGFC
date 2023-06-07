#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
gpu=$1
true="true"
echo "$gpu" 
if [[ $gpu == $true ]]; then
 echo GPU found
 docker run --name negfc_cnn -it \
    --rm \
    --privileged=true \
    --mount "type=bind,src=$(pwd),dst=/home/" \
    --workdir /home/ \
    -p 8888:8888 \
    -p 6006:6006 \
    --gpus all \
    -e HOST="$(whoami)" \
    negfc_cnn bash
else
 echo CPU mode
 docker run --name negfc_cnn -it \
   --rm \
   --privileged=true \
   --mount "type=bind,src=$(pwd),dst=/home/" \
   --workdir /home/ \
   -p 8888:8888 \
   -p 6006:6006 \
   -e HOST="$(whoami)" \
   negfc_cnn bash
fi
