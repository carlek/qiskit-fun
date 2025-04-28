# `qiskit-fun`

`grovers-euchre.ipynb`:  Grover's algorithm to find winning hands in Euchre. 

- https://carlek.hashnode.dev/quantum-search-and-euchre

- https://learning.quantum.ibm.com/tutorial/grovers-algorithm

### Building package `tweedledum`:

This package was not installable via PyPi for Apple Silicon.
Build from source... 
```bash
git clone git@github.com:boschmitt/tweedledum.git
cd tweedledum; mkdir build; cd build
CMAKE_PREFIX_PATH=$(python3 -m pybind11 --cmakedir) cmake -DTWEEDLEDUM_EXAMPLES=TRUE ..
make
cd ..
CMAKE_PREFIX_PATH=$(python3 -m pybind11 --cmakedir) pip wheel . -w dist
```
