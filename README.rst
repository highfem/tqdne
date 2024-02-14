tqdne: Demo packages with pytorch lightning
===========================================

Getting started
---------------

To get started, make sure you have the necessary permissions and clone the repository:

.. code-block:: bash

   git clone git@github.com:nperraud/tqdne.git
   cd tqdne

Installation
------------

To install all relevant dependencies, install conda and create a conda environment as shown below:

.. code-block:: bash
   
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Create a conda environment:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate tqdne

This will first install all dependencies and then the package.
If dependencies are missing or conflicting, please issue a bug report or file a PR with a fix.

Usage
-----

Change directory into `notebooks` and then run `jupyter`:

.. code-block:: bash

    cd notebooks
    jupyter lab


Linting and unit tests
----------------------

You can check lints and run unit tests using:

.. code-block:: bash
   
   tox
