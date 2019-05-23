#!/bin/sh
#g++ `pkg-config --cflags opencv` glasses.cpp -o plikwykonywalny `pkg-config --libs opencv`
a=$0;
b=${a%.*}
name=`echo $b | tr -d './'`

#g++ -O3 -std=c++0x -fopenmp `pkg-config --cflags opencv` $name.cpp -o $name `pkg-config --libs opencv`
g++ -std=c++14 -fopenmp `pkg-config --cflags opencv` $name.cpp -o $name `pkg-config --libs opencv`
