# HOGWILD Project Starter

This project is a baseline for studying stochastic gradient descent (SGD) and later extending it to parallel and lock-free variants such as HOGWILD.

## features atm
- Binary logistic regression
- Synthetic dataset generation
- Sequential SGD training
- Loss and accuracy reporting

## Build
```bash
mkdir build
cd build
cmake ..
make
./hogwild