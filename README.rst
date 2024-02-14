tqdne: Demo packages with pytorch lightning
===========================================

Getting started
---------------

To get started, make sure you have the necessary permissions and clone the repository:

.. code-block:: bash

   git clone git@github.com:nperraud/tqdne.git
   cd tqdne


You also need to install the submodule:

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
    jupyter notebooks

Tests
-----

The tests are located in the folder tqdne/tests. The tests are run using pytest. To run the tests, use the following command:

.. code-block:: bash

    make test


Documentation
-------------

Check the sphynx documentation in the folder doc. Update the documentation accordingly.

You can compile the doc using the following command:

.. code-block:: bash

   make docs


Style and linting
-----------------

The code is linted using flake8. To run the linter, use the following command:


.. code-block:: bash
   
   make lint
