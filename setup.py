import pathlib

import setuptools

# The directory containing this file
MODULE_NAME = "knobtimizer"
TOPLEVEL_DIR = pathlib.Path(__file__).parent.absolute()
ABOUT_FILE = TOPLEVEL_DIR / MODULE_NAME / "__init__.py"
README = TOPLEVEL_DIR / "README.md"


def about_package(init_posixpath: pathlib.Path) -> dict:
    """
    Return package information defined with dunders in __init__.py as a dictionary, when
    provided with a PosixPath to the __init__.py file.
    """
    about_text: str = init_posixpath.read_text()
    return {
        entry.split(" = ")[0]: entry.split(" = ")[1].strip('"')
        for entry in about_text.strip().split("\n")
        if entry.startswith("__")
    }


ABOUT_KNOBTIMIZER = about_package(ABOUT_FILE)

with README.open("r") as docs:
    long_description = docs.read()

# Dependencies for the module itself
DEPENDENCIES = [
    'dill',
    'tfs-pandas',
    'jinja2',
    'pytest',
    'pymoo',
    'generic-parser',
    'dask[complete]',
    'dask_jobqueue',
    'pyyaml',
]

EXTRA_DEPENDENCIES = {
    "test": [
        "pytest>=5.2",
        "pytest-cov>=2.7",
    ],
    "doc": ["sphinx", "sphinx_rtd_theme"],
}
EXTRA_DEPENDENCIES.update(
    {"all": [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
)


setuptools.setup(
    name=ABOUT_KNOBTIMIZER["__title__"],
    version=ABOUT_KNOBTIMIZER["__version__"],
    description=ABOUT_KNOBTIMIZER["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=ABOUT_KNOBTIMIZER["__author__"],
    author_email=ABOUT_KNOBTIMIZER["__author_email__"],
    url=ABOUT_KNOBTIMIZER["__url__"],
    python_requires=">=3.7",
    license=ABOUT_KNOBTIMIZER["__license__"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=setuptools.find_packages(exclude=["tests*", "doc"]),
    include_package_data=True,
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES["test"],
    extras_require=EXTRA_DEPENDENCIES,
)