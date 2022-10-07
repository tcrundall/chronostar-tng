import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="chronostar-trial",
    version="0.0.1",
    author="Timothy Crundall",
    author_email="tim.crundall@gmail.com",
    description=("An astrophysical tool for discovering and characterising"
                 "stellar associations."),
    license="BSD",
    keywords="astrophysics stellar associations",
    url="https://github.com/tcrundall/chronostar-tng",

    # packages=find_packages('src'),
    # package_dir={'': 'src'},

    packages=find_packages('src'),
    package_dir={'': 'src'},

    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    scripts=[
        'bin/fit-component',
        'bin/fit-mixture',
    ]
)
