"""
The EAGLE and SubFind frontend for :mod:``transfer``.

Requires the ``h5py`` python package.
"""

from transfer import LOGGER

from transfer.data import ParticleData, SnapshotData

from typing import Optional
from unyt import unyt_array, unyt_quantity
from math import pow
from numpy import full, concatenate

import h5py


class EAGLEParticleData(ParticleData):
    """
    EAGLE particle data frontend. Implements the :class:`ParticleData`
    class with real functionality for reading EAGLE snapshots.
    """

    def __init__(self, filename: str, particle_type: int, num_files: int):
        """
        Parameters
        ----------

        filename: str
            The EAGLE snapshot filename to extract the particle data from,
            without the ``.x.hdf5``.

        particle_type: int
            The particle type to load (0, 1, 4, etc.)

        num_files: int
            The number of files to read from (as EAGLE snapshots are often
            split into multiple files).
        """
        super().__init__()

        self.filename = filename
        self.particle_type = particle_type
        self.num_files = num_files

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

        raw, cgs = load_data("Coordinates")

        units = unyt_quantity(cgs, units="cm").to("Mpc")

        return unyt_array(raw, units=units)

    def load_masses(self):
        """
        Loads the masses from the file, returning an unyt array.
        """

        try:
            raw, cgs = load_data("Mass")
        except KeyError:
            # Somebody thought it was a good idea in EAGLE to remove the
            # Mass attribute from PartType1.

            with h5py.File(f"{self.filename}.0.hdf5", "r") as handle:
                part_mass = handle["Header"].attrs["MassTable"][self.particle_type]
                num_part = handle["Header"].attrs["NumPart_Total"][self.particle_type]
                hubble_param = handle["Header"].attrs["HubbleParam"]

                cgs = handle["Units"].attrs["UnitMass_in_g"] / hubble_param
                raw = full(num_part, fill_value=part_mass)

        units = unyt_quantity(cgs, units="g").to("Solar_Mass")

        return unyt_array(raw, units=units)

    def load_particle_ids(self):
        """
        Loads the Particle IDs from the file, returning an unyt array.
        """

        raw, _ = load_data("ParticleIDs")

        return unyt_array(raw, units="none")

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

        units: float
            Conversion to CGS units for this type, read from CGSConversionFactor.
            Includes conversion to remove h-factor.

        """

        full_path = f"/PartType{self.particle_type}/{array_name}"

        output = []

        with h5py.File(f"{self.filename}.0.hdf5", "r") as handle:
            units = handle[array_name].attrs["CGSConversionFactor"]
            h_exponent = handle[array_name].attrs["h-scale-exponent"]
            h_factor = pow(handle["Header"].attrs["HubbleParam"], h_exponent)
            units *= h_factor

        for file in range(self.num_files):
            current_filename = f"{self.filename}.{file}.hdf5"

            with h5py.File(current_filename, "r") as handle:
                output.append(handle[to_read][...])

        output = concatenate(output)

        return output, units

    def perform_particle_id_postprocessing(self):
        """
        Performs postprocessing on ParticleIDs to ensure that they link
        correctly (required in cases where the particle IDs are offset when
        new generations of particles are spawned).
        """

        LOGGER.info("Beginning particle ID postprocessing (empty).")

        return


class EAGLESnapshotData(SnapshotData):
    """
    EAGLE particle data frontend. Implements the :class:`SnapshotData`
    class with real functionality for reading EAGLE snapshots and
    EAGLE SubFind catalogues.
    """

    def __init__(
        self,
        filename: str,
        num_files_particles: int,
        halo_filename: Optional[str] = None,
        num_files_halo: Optional[int] = None,
    ):
        """
        Parameters
        ----------

        filename: str
            Filename for the snapshot file, without the .x.hdf5.

        num_files_particles: int
            Number of files that the snapshot file is split into.

        halo_filename: str, optional
            Filename for the halo file, without the .x.hdf5.

        num_files_halo: int, optional
            Number of files that the halo file is split into.
        """

        super().__init__(filename=filename, halo_filename=halo_filename)

        self.num_files_particles = num_files_particles
        self.num_files_halo = num_files_halo

        return

    def load_particle_data(self):
        """
        Loads the particle data from a snapshot using ``h5py``.
        """

        self.boxsize = data.metadata.boxsize

        for particle_type, particle_name in zip(
            [0, 1, 4], ["dark_matter", "gas", "stars"]
        ):
            setattr(
                self,
                particle_name,
                EAGLEParticleData(
                    filename=self.filename,
                    particle_type=particle_type,
                    num_files=self.num_files_particles,
                ),
            )

        return

    def load_halo_data(self):
        """
        Loads haloes from SubFind and, using the center of potential
        and R_200, uses trees through :meth:`ParticleDataset.associate_haloes`
        to set the halo values.
        """

        if self.halo_filename is None:
            return

        LOGGER.info(f"Loading halo catalogue data from {self.halo_filename}")

        # We want to extract the center of potential and r_200mean for central
        # galaxies only, based on their FoF group. However, based on this FOF
        # group we do not know which haloes are centrals. For this we need to
        # use the following:
        #
        #   Subhalo/Subgroupnumber == 0 gives centrals
        #   Subhalo/Groupnumber indexes the FOF catalogue
        #   FOF/Group_R_Mean200 gives the R_200mean
        #   FOF/GroupCentreOfPotential gives the center of potential

        # First load the length units out of the catalogue, and h-correct
        with h5py.File(f"{self.halo_filename}.0.hdf5", "r") as handle:
            hubble_param = handle["Header"].attrs["HubbleParam"]
            cgs = handle["Units"].attrs["UnitLength_in_cm"] / hubble_param
            boxsize = handle["Header"].attrs["BoxSize"]

        units = unyt_quantity(cgs, units="cm").to("Mpc")
        self.boxsize = unyt_quantity(boxsize, units=units)

        # FoF group info; this will be sliced
        center_of_potential = []
        r_200mean = []
        central_group_numbers = []

        for file in range(self.num_files_halo):
            current_filename = f"{self.halo_filename}.{file}.hdf5"

            with h5py.File(current_filename, "r") as handle:
                center_of_potential.append(handle["/FOF/GroupCentreOfPotential"][...])
                r_200mean.append(handle["/FOF/Group_R_Mean200"][...])

                sub_group_number = handle["/Subhalo/SubGroupNumber"][...]
                group_number = handle["/Subhalo/GroupNumber"][...]

                central_group_numbers.append(group_number[sub_group_number == 0])

        # We currently have a list of arrays, need to stick together
        central_group_numbers = concatenate(central_group_numbers)

        # This slicing removes all non-central FoF groups.
        center_of_potential = concatenate(center_of_potential)[central_group_numbers]
        r_200mean = concatenate(r_200mean)[central_group_numbers]

        halo_coordinates = unyt_array(center_of_potential, units=units)
        halo_radii = unyt_array(r_200mean, units=units)

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
