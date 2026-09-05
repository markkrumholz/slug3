.. highlight:: rest

Getting SLUG
==========================

SLUG v3 is available for download from the GitHub repository at
`https://github.com/markkrumholz/slug3 <https://github.com/markkrumholz/slug3>`_.
The easiest way to install SLUG is to use git to clone the repository by doing::

    git clone https://github.com/markkrumholz/slug3.git
    git submodule update --init --recursive

In addition to the SLUG source code and its submodules, SLUG requires a number of
data files that are not included in the github repository due to their size. These data
files can be downloaded using the script ``data/tools/download_data.py`` in the
repository. For example, to download all of the data files, run::

    python data/tools/download_data.py --all

Warning: the data files are large (several GB), so make sure you have enough disk space
before downloading them.

If you prefer to download the data files manually, they can be accessed from the
Australian National University
`Data Commons <https://datacommons.anu.edu.au/DataCommons/rest/display/anudc:6518>`_
service.