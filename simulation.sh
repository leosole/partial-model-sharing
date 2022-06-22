#!/bin/bash
for i in {1..15}
do
    echo "start split $i"
    python split_client.py 1 $1 &
    python split_client.py 2 $1 &
    wait
    echo "end split $i"
done

for i in {1..15}
do
    echo "start transfer $i"
    python transfer_client.py 1 $1 &
    python transfer_client.py 2 $1 &
    wait
    echo "end transfer $i"
done

for i in {1..15}
do
    echo "start local $i"
    python local_client.py 1 $1 &
    python local_client.py 2 $1 &
    wait
    echo "end local $i"
done

for i in {1..15}
do
    echo "start global $i"
    python global_client.py $1 &
    wait
    echo "end global $i"
done
