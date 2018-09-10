#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8

import os
import subprocess as sub

# check if this script is executed from the package root directory
assert(os.path.isfile("./scripts/update_git_hash.py"))

package_name = "bioen"

try:
    cmd = "git describe --all --long --dirty --tags".split()
    raw = sub.check_output(cmd).rstrip().split("/")[1]
except:
    raw = "n/a"

with open("./" + package_name + "/githash.py", "w") as fp:
    fp.write("human_readable = \"" + raw + "\"\n")
