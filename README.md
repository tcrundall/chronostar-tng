![unit tests](https://github.com/tcrundall/chronostar-tng/actions/workflows/unit-tests.yml/badge.svg)
![integration tests](https://github.com/tcrundall/chronostar-tng/actions/workflows/integration-tests.yml/badge.svg)
![flake8](https://github.com/tcrundall/chronostar-tng/actions/workflows/flake8.yml/badge.svg)
# Chronostar-TNG

Chronostar, the next generation of discovery and characterisation of stellar associations

[The full docs](https://tcrundall.github.io/chronostar-tng/) have a more detailed installation and quickstart.

### Installing
Create a conda environment and install packages available through conda.
```
git clone https://github.com/tcrundall/chronostar-tng.git
conda env create -n chron -f environment.yml    # set up py39 environment
conda activate chron
pip install .       # this will put chronostar-trial in your site-packages
```

You will now have three command-line tools at your disposal.
Prepare your data into a numpy array of shape `(n_stars, n_features)`,
where the features are in RHS cartesian coordinates centred on the local
standard of rest (`XYZUVW`).

In any directory you can call:
```
>>> fit-component path/to/data.npy [path/to/memberships.npy]
>>> fit-mixture NCOMPONENTS path/to/data.npy
    or
>>> fit-chronostar path/to/data.npy
```

Where `memberships.npy` is a stored numpy array of shape `(n_stars)` with
entries between `0` and `1`.
