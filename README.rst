tqdne: Demo packages with pytorch lightning
=============================================


Getting started
---------------

To get started, make sure you have the necessary permissions and clone the repository:

.. code-block:: bash

   git clone git@github.com:nperraud/tqdne.git
   cd tqdne


You also need to install the submodule:

.. code-block:: bash

   git submodule init
   git submodule update


Working with Poetry
-------------------

To install Poetry, run the following command in the terminal:

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3 -

Activate the Poetry shell and install the dependencies:

.. code-block:: bash

   poetry shell
   poetry install

Working with Conda
------------------

To install miniconda, run the following command in the terminal:

.. code-block:: bash
   
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      bash Miniconda3-latest-Linux-x86_64.sh

Create a conda environment:

.. code-block:: bash

   conda env create -f environment.yaml
   conda activate tqdne



Tests
-----

The tests are located in the folder tqdne/tests. The tests are run using pytest. To run the tests, use the following command:

.. code-block:: bash

   pytest tqdne

or

.. code-block:: bash
   
   make test



Documentation
-------------

Check the sphynx documentation in the folder doc. Update the documentation accordingly.

You can compile the doc using the following command:

.. code-block:: bash

   make doc



Style and linting
-----------------

The code is linted using flake8. To run the linter, use the following command:

.. code-block:: bash

   flake8 --doctests --exclude=doc --ignore=E501

or

.. code-block:: bash
   
   make lint


To help you to get the right format, you can use `black`:

.. code-block:: bash

   black tqdne
