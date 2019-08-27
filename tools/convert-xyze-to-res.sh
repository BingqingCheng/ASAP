#!/bin/bash

xyzefile=$1
prefix=${xyzefile%.*}

cabal xyze res < ${xyzefile} > ${prefix}.res
