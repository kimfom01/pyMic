#!/usr/bin/python

from __future__ import print_function

import pymic as mic
import numpy as np
import sys
import time

# load the library with the kernel function (on the target)
device = mic.devices[0]
library = device.load_library("matrix_multiplication.so")

# use the default stream
stream = device.get_default_stream()

ds = 10
m, n, k = ds, ds, ds
if len(sys.argv) > 1:
    sz = int(sys.argv[1])
    m, n, k = sz, sz, sz

# construct some matrices
np.random.seed(10)
a = np.random.random(m * k).reshape((m, k))
b = np.random.random(k * n).reshape((k, n))
c = np.zeros((m, n))

# associate host arrays with device arrats
offl_a = stream.bind(a)
offl_b = stream.bind(b)
offl_c = stream.bind(c)

# convert a and b to matrices (eases MxM in numpy)
Am = np.matrix(a)
Bm = np.matrix(b)

# print the input
print("input:")
print("--------------------------------------")
print("a=", a)
print("b=", b)
print()
print()

# print the input of numpy's MxM if it is small enough
np_mxm_start = time.time()
Cm = Am * Bm
np_mxm_end = time.time()
print("numpy gives us:")
print("--------------------------------------")
print(Cm)
print("checksum:", np.sum(Cm))
print()
print()

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# invoke the offloaded dgemm
c[:] = 0.0
offl_c.update_device()
np_mic_start = time.time()
stream.invoke(library.multiplication, offl_a, offl_b, offl_c, m,k)
stream.sync()
np_mic_end = time.time()


offl_c.update_host()
stream.sync()
print(offl_c)
