"""
Lagrangian Transfer module (v2.0).

This is a complete re-write of the module that was used for the original
paper, as that was a) somewhat inefficient and b) had an overly complicated
strucutre that made modifications difficult.

Josh Borrow (joshua.borrow@durham.ac.uk) (2020).
"""

import logging

LOGGER = logging.getLogger("LagrangianTransferLogger")
LOGGER.setLevel(10)
