"""
The SWIFT and VELOCIraptor frontend for :mod:``transfer``.

Requires the ``swiftsimio`` and ``velociraptor`` python packages.
"""

from transfer import LOGGER

from velociraptor import load as load_catalogue
from swiftsimio import load as load_snapshot

from transfer.data import ParticleData, SnapshotData

from typing import Optional
from unyt import unyt_array


class SWIFTParticleData(ParticleData):
    """
    SWIFT particle data frontend. Implements the :class:`ParticleData`
    class with real functionality for reading SWIFT snapshots.
    """

    def __init__(self, SWIFTDataset):
        """
        Parameters
        ----------

        SWIFTDataset
            The SWIFT dataset (e.g. ``x.dark_matter``) to extract
            the particle data from.
        """
        super().__init__()

        LOGGER.info(f"Loading particle data from {SWIFTDataset}")
        self.coordinates = SWIFTDataset.coordinates
        self.masses = SWIFTDataset.masses
        self.particle_ids = SWIFTDataset.particle_ids
        LOGGER.info(f"Finished loading data from {SWIFTDataset}")
        LOGGER.info(f"Loaded {self.particle_ids.size} particles")

        self.perform_particle_id_postprocessing()

        return

    def perform_particle_id_postprocessing(self):
        """
        Performs postprocessing on ParticleIDs to ensure that they link
        correctly (required in cases where the particle IDs are offset when
        new generations of particles are spawned).
        """

        LOGGER.info("Beginning particle ID postprocessing (empty).")

        return


class SWIFTSnapshotData(SnapshotData):
    """
    SWIFT particle data frontend. Implements the :class:`SnapshotData`
    class with real functionality for reading SWIFT snapshots.
    """

    def __init__(self, filename: str, halo_filename: Optional[str]):
        super().__init__(filename=filename, halo_filename=halo_filename)

        return

    def load_particle_data(self):
        """
        Loads the particle data from a snapshot using ``swiftsimio``.
        """

        data = load_snapshot(self.filename)

        self.boxsize = data.metadata.boxsize

        for particle_type in ["dark_matter", "gas", "stars"]:
            swift_dataset = getattr(data, particle_type, None)

            if swift_dataset is not None:
                setattr(self, particle_type, SWIFTParticleData(swift_dataset))
            else:
                LOGGER.info(f"No particles of type {particle_type} in {self.filename}")

        return

    def load_halo_data(self):
        """
        Loads haloes from VELOCIraptor and, using the most bound particle center
        and R_200, uses trees through :meth:`SWIFTParticleDataset.associate_haloes`
        to set the halo values.
        """

        if self.halo_filename is None:
            return

        LOGGER.info(f"Loading halo catalogue data from {self.halo_filename}")

        catalogue = load_catalogue(self.halo_filename)

        # Select only centrals
        centrals = catalogue.structure_type.structuretype == 10
        self.number_of_groups = centrals.sum()

        halo_coordinates = (
            unyt_array(
                [
                    getattr(catalogue.positions, f"{x}cmbp")[centrals]
                    for x in ["x", "y", "z"]
                ]
            ).T
            / catalogue.units.a
        )

        halo_radii = catalogue.radii.r_200mean[centrals] / catalogue.units.a

        LOGGER.info("Finished loading halo catalogue data")

        for particle_type in ["dark_matter", "gas", "stars"]:
            particle_data = getattr(self, particle_type, None)

            if particle_data is not None:
                LOGGER.info(f"Associating haloes for {particle_type}")
                particle_data.associate_haloes(
                    halo_coordinates=halo_coordinates,
                    halo_radii=halo_radii,
                    boxsize=self.boxsize,
                )

        return

    def sort_all_data(self):
        """
        Sorts all data currently present in the snapshot by particle ID on
        a particle type by particle type basis.
        """

        for particle_type in ["dark_matter", "gas", "stars"]:
            particle_data = getattr(self, particle_type, None)

            if particle_data is not None:
                particle_data.sort_by_particle_id()

        return
