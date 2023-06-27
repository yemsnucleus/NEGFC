#!/bin/bash
USER0="$1"
PORT_0="$2"
PORT_1=$((PORT_0 + 1))
echo "Cleaning up ssh connections associated with $USER0"
ssh $USER0@www.astro.udp.cl  "pkill -u $USER0 ssh"

echo "$PORT_0 (local) <-- $PORT_1 (Hendrix) <-- $PORT_0 (Leia)"
ssh -f $USER0@www.astro.udp.cl -L $PORT_0:localhost:$PORT_1 -N
PDI_0=$(pgrep -n ssh -f)
ssh $USER0@www.astro.udp.cl "ssh -f $USER0@172.16.101.28 -L $PORT_1:localhost:$PORT_0 -N" &
echo "Tunel successfully created"