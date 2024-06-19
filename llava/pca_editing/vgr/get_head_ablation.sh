#!/bin/bash
for n in {10,50,100,300,500,1000,2000,5000,10000}
do
    python get_head_values.py --n-samples $n
done