#!/bin/bash

scons parallelization=cuda solver=fwave cudaSDKDir=/ compileMode=debug
sleep 1
./build/SWE_gnu_debug_cuda_fwave 320 320 ./output
