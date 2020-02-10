"""
Contains the core jit-ified functions for computing transfer.
"""

from numpy import zeros, array
from numba import njit, jitclass, int32, int64, float64

from typing import Tuple

transfer_output_spec = [
    ("in_halo", float64[:]),
    ("in_halo_from_own_lr", float64[:]),
    ("in_halo_from_other_lr", float64[:]),
    ("in_halo_from_outside_lr", float64[:]),
    ("in_lr", float64[:]),
    ("in_other_halo_from_lr", float64[:]),
    ("outside_haloes", float64[:]),
]


@jitclass(transfer_output_spec)
class TransferOutput(object):
    """
    Output of the :func:`calculate_transfer_masses` function.

    Has the following attributes:

    Attributes
    ----------

    in_halo: np.array[float64]
        Mass in each halo.

    in_halo_from_own_lr: np.array[float64]
        Mass in own halo originating from the same Lagrangian Region.

    in_halo_from_other_lr: np.array[float64]
        Mass in this halo originating from anoter halo's Lagrangian Region.

    in_halo_from_outside_lr: np.array[float64]
        Mass in this halo originating outside any Lagrangian Region.


    in_lr: np.array[float64]
        Mass originating in each Lagrangian Region.

    in_other_halo_from_lr: np.array[float64]
        Mass from this Lagrangian Region that has ended up in another halo.

    outside_haloes: np.array[float64]
        Mass from this Lagrangian Region that has ended up outside any halo.

    """

    def __init__(
        self,
        in_halo,
        in_halo_from_own_lr,
        in_halo_from_other_lr,
        in_halo_from_outside_lr,
        in_lr,
        in_other_halo_from_lr,
        outside_haloes,
    ):
        """
        You should never create an instance of this class yourself. It should
        only be created inside :func:`calculate_transfer_masses`.
        """

        self.in_halo = in_halo
        self.in_halo_from_own_lr = in_halo_from_own_lr
        self.in_halo_from_other_lr = in_halo_from_other_lr
        self.in_halo_from_outside_lr = in_halo_from_outside_lr

        self.in_lr = in_lr
        self.in_other_halo_from_lr = in_other_halo_from_lr
        self.outside_haloes = outside_haloes

        return


@njit(parallel=True)
def calculate_transfer_masses(
    haloes: int32,
    lagrangian_regions: int32,
    particle_masses: float64,
    initial_particle_mass: float64,
    number_of_groups: int32,
) -> TransferOutput:
    """
    Calculates the transfer masses comparing lagrangian region IDs and
    halo IDs. Works as a linear pass over particles.

    Parameters
    ----------

    haloes: np.array[int64]
        Halo IDs for all particles. We assume that these run from
        0 to ``number_of_groups``. Particles outside any halo should
        have halo ID = -1.

    lagrangian_regions: np.array[int64]
        Lagrangian Region IDs for all particles. We assume that
        these run from 0 to ``number_of_groups``. Particles outside
        any LR should have ID = -1.

    particle_masses: np.array[float64]
        Final state particle mass for all particles.

    initial_particle_mass: float64
        Initial particle mass (assumed to be the same for all
        particles).

    number_of_groups: int64
        Total number of groups in the simulation.


    Returns
    -------

    output: TransferOutput
        Instance of :class:`TransferOutput` containing output arrays of
        size ``number_of_groups``. See the documentation for the class
        for more information.


    Notes
    -----

    The return arrays are all of size ``number_of_groups``. Relative to the
    version 1.0 of the LT code, this function crashses if halo IDs or
    LR IDs lead to out-of-bounds behaviour, instead of skipping.
    """

    number_of_particles = len(lagrangian_regions)

    # Output mass arrays
    in_halo = zeros(number_of_groups, dtype=float64)
    in_halo_from_own_lr = zeros(number_of_groups, dtype=float64)
    in_halo_from_other_lr = zeros(number_of_groups, dtype=float64)
    in_halo_from_outside_lr = zeros(number_of_groups, dtype=float64)

    in_lr = zeros(number_of_groups, dtype=float64)
    in_other_halo_from_lr = zeros(number_of_groups, dtype=float64)
    outside_haloes = zeros(number_of_groups, dtype=float64)

    for particle in range(number_of_particles):
        id = haloes[particle]
        lr = lagrangian_regions[particle]
        mass = particle_masses[particle]

        # Logic all collected here
        particle_in_halo = id != -1
        particle_in_lr = lr != -1
        particle_in_same = id == lr and particle_in_halo
        particle_in_other = id != lr and particle_in_halo and particle_in_lr
        particle_from_outside = particle_in_halo and not particle_in_lr
        particle_now_outside = particle_in_lr and not particle_in_halo

        # Propagate masses back to arrays
        in_halo[id] += mass if particle_in_halo else 0.0
        in_halo_from_own_lr[id] += mass if particle_in_same else 0.0
        in_halo_from_other_lr[id] += mass if particle_in_other else 0.0
        in_halo_from_outside_lr[id] += mass if particle_from_outside else 0.0

        in_lr[lr] += initial_particle_mass if particle_in_lr else 0.0
        in_other_halo_from_lr[lr] += initial_particle_mass if particle_in_other else 0.0
        outside_haloes[lr] += initial_particle_mass if particle_now_outside else 0.0

    # Only accepts non-keyword arguments - be careful of ordering!
    output = TransferOutput(
        in_halo,
        in_halo_from_own_lr,
        in_halo_from_other_lr,
        in_halo_from_outside_lr,
        in_lr,
        in_other_halo_from_lr,
        outside_haloes,
    )

    return output
