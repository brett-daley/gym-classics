#!/bin/bash
for subdir in gym gymnasium
do
    python -m unittest discover -s tests/$subdir -p "test_*.py"
done
