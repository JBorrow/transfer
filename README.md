Lagrangian Transfer 2.0
=======================

This repository contains a complete re-write of the Lagrangian Transfer
code in [this repository](https://github.com/JBorrow/lagrangian-transfer/).

The decision was made to re-write the code to provide simultaneously
faster speeds and cleaner code. The new code has a more modular design
and allows for the use of `numba` in tight loops. Things are also significantly
better laid out and contain `numpy`-style docstrings.

There is developer documentation available in `docs`.
