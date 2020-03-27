from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "1.2.5"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

setup(
    name="h5imagegenerator",
    version=__version__,
    description="A dead simple Keras HDF5 ImageDataGenerator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/angulartist/Keras-HDF5-ImageDataGenerator",
    download_url="https://github.com/angulartist/Keras-HDF5-ImageDataGenerator/tarball/"
    + __version__,
    license="BSD",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="",
    packages=find_packages(exclude=["doc", "tests*", "playground", "examples"]),
    include_package_data=True,
    author="@angulartist",
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email="michel@kebab.io",
    python_requires=">=3",
)
