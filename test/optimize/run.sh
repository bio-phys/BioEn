#!/bin/bash

# Ignore the deprecation warnings from the newest NumPy,
# caused by pickle files still containing matrix objects.

py.test -Wignore::PendingDeprecationWarning -sv

