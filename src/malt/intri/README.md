### `malt` submodule for geometry processing with Intrinsic Triangulations

Intrinsic triangulations are a powerful technique for computing with 3D
surfaces. Among other things, they enable existing algorithms to work "out
of the box" on poor-quality triangulations. The basic idea is to represent
the geometry of a triangle mesh by edge lengths, rather than vertex
positions; this change of perspective unlocks many powerful algorithms with
excellent robustness to poor-quality triangulations.

This course gives an overview of intrinsic triangulations and their use in
geometry processing, beginning with a general introduction to the basics and
historical roots, then covering recent data structures for encoding intrinsic
triangulations, and their application to tasks in surface geometry ranging
from geodesics to vector fields to parameterization.

This course was presented at SIGGRAPH 2021 and IMR 2021.

Authors: Nicholas Sharp, Mark Gillespie, Keenan Crane
https://github.com/nmwsharp/intrinsic-triangulations-tutorial

Modified by Max Eschenbach, DDU, TU Darmstadt