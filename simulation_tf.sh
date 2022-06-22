#!/bin/bash
for i in {1..10}
do
    echo "start split $i"
    python tf_test_partial.py 1 &
    python tf_test_partial.py 2 &
    wait
    echo "end split $i"
done

for i in {1..10}
do
    echo "start transfer $i"
    python tf_test_transfer.py 1 &
    python tf_test_transfer.py 2 &
    wait
    echo "end transfer $i"
done

for i in {1..10}
do
    echo "start local $i"
    python tf_test_local.py 1 &
    python tf_test_local.py 2 &
    wait
    echo "end local $i"
done

for i in {1..10}
do
    echo "start global $i"
    python tf_test_global.py &
    wait
    echo "end global $i"
done
