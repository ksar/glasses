#!/bin/sh
#g++ `pkg-config --cflags opencv` glasses.cpp -o plikwykonywalny `pkg-config --libs opencv`
a=$0;
b=${a%.*}
name=`echo $b | tr -d './'`

g++ -std=c++0x -fopenmp `pkg-config --cflags opencv` $name.cpp -o $name `pkg-config --libs opencv`
