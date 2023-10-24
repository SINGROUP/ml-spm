#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
cd $script_dir/mlspm/_c

g++ -fPIC -O3 -c matching.cpp peaks.cpp &&
g++ -shared matching.o peaks.o -o mlspm_lib.so
