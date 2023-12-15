# int4matmul

## Environmental Requirement

You need to make sure the machine is properly driven.

* sudo apt-get install clinfo
* sudo apt-get install ocl-icd-opencl-dev

* sudo apt-get install ocl-icd-dev

You can use clinfo to view information about your graphics card.

* clinfo

## Sample Purpose

This test case tests matrix multiplication, using the int4 data type.

* C(M, N) = A(M, K) * B(K, N)

matrix size: 

* M = 16
* K = 16
* N = 16

## How to Build the Samples

For example:

    make

    ./test

## Command Line Options

| Option         | Default Value | Description                                                                         |
| :------------- | :-----------: | :---------------------------------------------------------------------------------- |
| `-d <index>` |       0       | Specify the index of the OpenCL device in the platform to execute on the sample on. |
| `-p <index>` |       0       | Specify the index of the OpenCL platform to execute the sample on.                  |
