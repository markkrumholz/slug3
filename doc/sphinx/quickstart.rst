.. highlight:: rest

Quickstart
==========

To get started with SLUG, carry out the following steps:

1. Download SLUG and its required data by doing:

    .. code-block:: bash
    
        git clone https://github.com/markkrumholz/slug3.git
        git submodule update --init --recursive
        python data/tools/download_data.py --all

    See :ref:`sec-getting` for full details.

2. Compile SLUG by doing:

    .. code-block:: bash

        cmake -S . -B build -G Ninja
        cmake --build build

    from the repository root. See :ref:`sec-building` for full details. (If this
    doesn't work, check the list of dependencies and supported compilers in
    :ref:`sec-building`.)

3. Run the ``quick`` test suite to make sure everything is working:

    .. code-block:: bash

        cd build
        ctest -L quick

    See :ref:`sec-tests` for the full details on the test suite.