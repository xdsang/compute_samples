## Code Structure

```
README.md               This file
LICENSE                 License information
CMakeLists.txt          Top-level CMakefile
external/               External Projects (headers and libs)
include/                Include Files (OpenCL C++ bindings)
samples/                Sample Applications
tutorials/              Tutorials
```

## How to Build the Samples

For example:
    git submodule update --init --recursive
    mkdir build && cd build
    cmake ..
    make

Then, build with the generated build files.

\* Other names and brands may be claimed as the property of others.
