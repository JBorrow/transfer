"""
The SIMBA and AHF frontend for :mod:``transfer``.

Requires the ``h5py`` python package.
"""

from transfer import LOGGER

from transfer.data import ParticleData, SnapshotData

from typing import Optional
from unyt import unyt_array, unyt_quantity
from numpy import full, concatenate, isin, genfromtxt, array, uint64, unique

import h5py

# Units without the h-correction.
unit_mass = unyt_quantity(1e10, "Solar_Mass")
unit_length = unyt_quantity(1.0, "kpc")
unit_velocity = unyt_quantity(1.0,"km/s")


class SIMBAParticleData(ParticleData):
    """
    SIMBA particle data frontend. Implements the :class:`ParticleData`
    class with real functionality for reading EAGLE snapshots.
    """

    def __init__(self, filename: str, particle_type: int, truncate_ids: Optional[int] = None):
        """
        Parameters
        ----------

        filename: str
            The SIMBA snapshot filename to extract the particle data from.

        particle_type: int
            The particle type to load (0, 1, 4, etc.)

        truncate_ids: int, optional
            Truncate IDs above this by using the % operator; i.e. discard
            higher bits.
        """
        super().__init__()

        self.filename = filename
        self.particle_type = particle_type
        self.truncate_ids = truncate_ids

        LOGGER.info(f"Loading particle data from particle type {particle_type}")
        self.coordinates = self.load_coordinates()
        self.masses = self.load_masses()
        self.particle_ids = self.load_particle_ids()
        LOGGER.info(f"Finished loading data from particle type {particle_type}")
        LOGGER.info(f"Loaded {self.particle_ids.size} particles")

        self.perform_particle_id_postprocessing()

        return

    def load_coordinates(self):
        """
        Loads the coordinates from the file, returning an unyt array.
        """

        raw, h = self.load_data("Coordinates")

        units = unyt_quantity(1.0 / h, units=unit_length).to("Mpc")

        return unyt_array(raw, units=units)

    def load_masses(self):
        """
        Loads the masses from the file, returning an unyt array.
        """

        raw, h = self.load_data("Masses")

        units = unyt_quantity(1.0 / h, units=unit_mass).to("Solar_Mass")

        return unyt_array(raw, units=units)

    def load_particle_ids(self):
        """
        Loads the Particle IDs from the file, returning an unyt array.
        """

        raw, _ = self.load_data("ParticleIDs")

        return unyt_array(raw.astype(uint64), units=None, dtype=uint64)

    def load_data(self, array_name: str):
        """
        Loads an array and returns it.

        Parameters
        ----------

        array_name: str
            Name of the array (without particle type) to read, e.g. Coordinates


        Returns
        -------

        output: np.array
            Output read from the HDF5 file

        h: float
            Hubble parameter.

        """

        full_path = f"/PartType{self.particle_type}/{array_name}"

        LOGGER.info(f"Loading data from {full_path}.")

        with h5py.File(f"{self.filename}", "r") as handle:
            h = handle["Header"].attrs["HubbleParam"]
            output = handle[full_path][:]

        return output, h

    def perform_particle_id_postprocessing(self):
        """
        Performs postprocessing on ParticleIDs to ensure that they link
        correctly (required in cases where the particle IDs are offset when
        new generations of particles are spawned).
        """

        if self.truncate_ids is None:
            LOGGER.info("Beginning particle ID postprocessing (empty).")
        else:
            LOGGER.info("Beginning particle ID postprocessing.")
            LOGGER.info(f"Truncating particle IDs above {self.truncate_ids}")

            self.particle_ids %= self.truncate_ids
            
            # TODO: Remove this requiremnet. At the moment, isin() breaks when
            #       you have repeated values.

            self.particle_ids, indicies = unique(self.particle_ids, return_index=True)
            self.coordinates = self.coordinates[indicies]
            self.masses = self.masses[indicies]

        return


class SIMBASnapshotData(SnapshotData):
    """
    SIMBA particle data frontend. Implements the :class:`SnapshotData`
    class with real functionality for reading SIMBA snapshots and
    SIMBA AHF catalogues.
    """

    def __init__(
            self, filename: str, halo_filename: Optional[str] = None, truncate_ids: Optional[dict] = None,
    ):
        """
        Parameters
        ----------

        filename: str
            Filename for the snapshot file.

        halo_filename: str, optional
            Filename for the halo file.

        truncate_ids: Dict[int,int], optional
            Dictionary from particle_type (i.e. 0, 1, 4): truncation 
            amount.

            Truncate IDs above this by using the % operator; i.e. discard
            higher bits.
        """

        self.load_boxsize(filename=filename)
        self.truncate_ids = truncate_ids

        super().__init__(filename=filename, halo_filename=halo_filename)

        return

    def load_boxsize(self, filename: str):
        """
        Loads the boxsize and hubble param from the particle file.
        """

        with h5py.File(filename, "r") as handle:
            hubble_param = handle["Header"].attrs["HubbleParam"]
            corrected_unit_length = unit_length / hubble_param
            boxsize = handle["Header"].attrs["BoxSize"]

        self.hubble_param = hubble_param
        self.boxsize = unyt_quantity(boxsize, units=corrected_unit_length).to("Mpc")

        return

    def load_particle_data(self):
        """
        Loads the particle data from a snapshot using ``h5py``.
        """

        for particle_type, particle_name in zip(
            [1, 0, 4], ["dark_matter", "gas", "stars"]
        ):
            truncate_ids = self.truncate_ids[particle_type] if self.truncate_ids is not None else None

            try:
                setattr(
                    self,
                    particle_name,
                    SIMBAParticleData(
                        filename=self.filename, particle_type=particle_type, truncate_ids=truncate_ids
                    ),
                )
            except KeyError:
                # No particles of this type (e.g. stars in ICs)
                LOGGER.info(
                    (
                        f"No particles of type {particle_type} ({particle_name}) "
                        "in this file. Skipping."
                    )
                )

                setattr(self, particle_name, None)

        return

    def load_halo_data(self):
        """
        Loads haloes from AHF and, using the center of mass of the halo
        and R_vir, uses trees through :meth:`ParticleDataset.associate_haloes`
        to set the halo values.

        Loads only central haloes (with hostHalo = -1)
        """

        if self.halo_filename is None:
            return

        LOGGER.info(f"Loading halo catalogue data from {self.halo_filename}")

        raw_data = genfromtxt(self.halo_filename, usecols=[1, 5, 6, 7, 11]).T

        hostHalo = raw_data[0].astype(int)
        mask = hostHalo == -1

        xmbp = raw_data[1][mask]
        ymbp = raw_data[2][mask]
        zmbp = raw_data[3][mask]

        center_of_potential = array([xmbp, ymbp, zmbp]).T
        r_vir = raw_data[4][mask]

        units = unyt_quantity(1.0 / self.hubble_param, units=unit_length).to("Mpc")

        halo_coordinates = unyt_array(center_of_potential, units=units)
        halo_radii = unyt_array(r_vir, units=units)

        self.number_of_groups = halo_radii.size

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
