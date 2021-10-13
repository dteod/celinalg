#!/usr/bin/bash -x

pushd `dirname ${BASH_SOURCE[0]}`
gcc $1 -o ${1%.*} -I ../include/public -I ../include/private -I ../../../utils/include/ -std=c++20 
popd