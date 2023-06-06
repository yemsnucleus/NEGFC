#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
echo 'Looking for GPUs (ETA: 10 seconds)'
gpu=$(lspci | grep -i '.* VGA .*')
shopt -s nocasematch

if [[ $gpu == * ]]; then
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
