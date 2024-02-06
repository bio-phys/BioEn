# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Version number and helper routines.

This module contains the version number at a single central location.
"""


ver = (0, 1, 3)


def get_package_name():
    """
    Returns name of package

    Returns
    -------
    string: name of package
    """
    return "bioen"


def get_version_string():
    """
    Returns version

    Returns
    -------
    string: version number

    """
    ###"""Return the full version number."""
    return '.'.join(map(str, ver))


def get_short_version_string():
    """
    Returns version number without the patchlevel

    Returns
    -------
    string: short version number

    """
    return '.'.join(map(str, ver[:-1]))


def get_printable_version_string():
    """
    Returns a formatted version of package name and version

    Returns
    -------
    string: versoin of the package name and version

    """
    ###"""Return a nicely formatted version string, if possible with git hash."""
    version_string = get_package_name() + " " + get_version_string()
    try:
        from . import githash
    except:
        pass
    else:
        version_string += " (git: " + githash.human_readable + ")"
    return version_string
