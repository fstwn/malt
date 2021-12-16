# Interface to the executables of the reference implementation of the paper
# "Rotation Invariant Spherical Harmonic Representation of 3D Shape
# Descriptors" by Michael Kazhdan, Thomas Funkhouser, and Szymon Rusinkiewicz
# Eurographics Symposium on Geometry Processing (2003)
#
# For reference see:
# https://www.cs.jhu.edu/~misha/MyPapers/SGP03.pdf
# https://github.com/mkazhdan/ShapeSPH
#
# licensed under MIT License
# Python interface written by Max Eschenbach, DDU, TU-Darmstadt, 2021

from .shapedescriptor import (compute_descriptor) # NOQA401
