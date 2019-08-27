#!/bin/bash

xyzefile=$1
prefix=${xyzefile%.*}

cabal xyz res < ${xyzefile} > ${prefix}.res
