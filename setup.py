import setuptools
from transfer import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transfer",
    version=__version__,
    description="Lagrangian Transfer",
    url="https://github.com/swiftsim/lagrangian-transfer-v2",
    author="Josh Borrow",
    author_email="joshua.borrow@durham.ac.uk",
    packages=setuptools.find_packages(),
    scripts=[],
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "unyt>=2.3.0"],
)
