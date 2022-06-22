#!/bin/bash
for i in {1..15}
do
    echo "start split $i"
    python split.py 1 $1 &
    python split.py 2 $1 &
    wait
    echo "end split $i"
done

for i in {1..15}
do
    echo "start local $i"
    python local.py 1 $1 &
    python local.py 2 $1 &
    wait
    echo "end local $i"
done

for i in {1..15}
do
    echo "start global $i"
    python global.py $1 &
    wait
    echo "end global $i"
done
