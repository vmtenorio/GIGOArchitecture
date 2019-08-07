#!/bin/sh
# Script that runs the python code passed as first argument each 5 seconds

while :
do
    python3 $1
    sleep 5
done

