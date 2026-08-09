"""
read.py

Implements read, a thin convenience wrapper around slug_reader.
"""

from .slug_reader import slug_reader


def read(filename: str) -> slug_reader:
    """
    Open a slug HDF5 output file for lazy reading.

    Parameters
    ----------
    filename : str
        Path to the slug HDF5 output file to open.

    Returns
    -------
    slug_reader
        A lazy reader for filename.
    """
    return slug_reader(filename)
