# for providing a descriptor of the surface mesh(esp. the triangle mesh)
# refer to:
#     Description of 3D-shape using a complex function on the sphere
#     D.V. Vranic, D. Saupe, 2002
#
# Script by GitHub User ReNicole
# https://github.com/ReNicole/ShapeDescriptor
#
# Modified by Max Eschenbach, DDU, TU-Darmstadt

# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import numpy as np
from scipy.interpolate import griddata
from pyshtools.expand import (SHExpandDH,
                              SHExpandDHC
                              )


# LOCAL MODULES IMPORTS -------------------------------------------------------

from .sample import triangle_face_sample
from .geometry import (get_trimesh_centroid,
                       get_trimesh_volume,
                       xyz2sp,
                       )


def uniform_volume(vertices, facet, reset_volume=1.):
    """
    reset the volume uniformly so that the volume of the different mesh will
    be the same.
    """
    src_volume = get_trimesh_volume(vertices, facet)
    scale = (src_volume / reset_volume) ** (1./3)
    vertices = vertices / scale
    return vertices, facet


def gridSampleXU(vertices, facet):
    """
    get samples of x(u) of the given mesh and make a grid form of the sample
    for the convenience to apply discrete spherical harmonics transform
    .. note: the definition of x(u) can be seen in func::descriptorRS
    output::grid_xu: numpy array (180,360)
    """
    # to generate the random samples for each degree in (-180,180),(-90,90)
    # the result is (n,3) numpy array, each row is the cartesian coordinate of
    # sample point
    # samples = trimesh.sample.sample_surface(mesh,360*180*3)
    vertices, facet = uniform_volume(vertices, facet)
    samples = triangle_face_sample(vertices, facet, number=(360*180*3))[0]
    # get aligned
    # samples = samples - np.array([mesh.center_mass]*samples.shape[0])
    centroid = get_trimesh_centroid(vertices, facet)
    samples = samples - np.array([centroid] * samples.shape[0])
    # convert the cartesian coordinate to spherical coordinate (and the first
    # coordinate radius is the desired x(u))
    spsam = xyz2sp(samples)
    # use nearest point on each grid position as the value and generate the
    # grid form of the sample value
    xi = np.linspace(-np.pi, np.pi, 360)
    yi = np.linspace(-0.5*np.pi, 0.5*np.pi, 180)
    grid_xu = griddata(spsam[:, 1:3], spsam[:, 0], (xi[None, :], yi[:, None]),
                       method='nearest')
    # xi,yi = np.mgrid[-np.pi:np.pi:360j, -0.5*np.pi:0.5*np.pi:180j]
    # grid_xu = griddata(spsam[:,1:3], spsam[:,0], (xi,yi), method='nearest')
    return grid_xu


def gridSampleYU(vertices, facet):
    """
    get samples of y(u) of the given mesh and make a grid form of the sample
    for the convenience to apply discrete spherical harmonics transform
    .. note: the definition of y(u) can be seen in func::descriptorSS
    ---------------------------------------------------------------------------
    output::grid_yu: numpy array (180,360)
    """
    # sample
    vertices, facet = uniform_volume(vertices, facet)
    samples, face_index = triangle_face_sample(vertices, facet,
                                               number=(360*180*3))
    # align
    centroid = get_trimesh_centroid(vertices, facet)
    samples = samples - np.array([centroid]*samples.shape[0])
    # get the cross product of (v2-v0),(v1-v0) of each face corresponding
    # to the sample points
    tri_cross = np.array([np.cross((vertices[facet[k, 2]]
                                    - vertices[facet[k, 0]]),
                                   (vertices[facet[k, 1]]
                                    - vertices[facet[k, 0]]))
                         for k in range(len(facet))])
    sample_cross = tri_cross[face_index]
    # get the norm of each cross product
    sample_corssnorm = np.linalg.norm(sample_cross, axis=1)
    # get y(u) for each sample point (without normalization)
    yu = np.array([np.dot(samples[k], sample_cross[k])
                   for k in range(len(samples))])
    # normalize
    yu = np.divide(yu, sample_corssnorm)
    # make sure y(u) is non-negative value(since we suppose dot(u,n(u)) should
    # non-negtive)
    yu = np.abs(yu)
    # convert the cartesian coordinate to spherical coordinate
    spsam = xyz2sp(samples)
    # use nearest point on each grid position as the value and generate the
    # grid form of the sample value y(u)
    xi = np.linspace(-np.pi, np.pi, 360)
    yi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 180)
    grid_yu = griddata(spsam[:, 1:3], yu, (xi[None, :], yi[:, None]),
                       method='nearest')
    # xi,yi = np.mgrid[-np.pi:np.pi:360j, -0.5*np.pi:0.5*np.pi:180j]
    # grid_yu = griddata(spsam[:,1:3], yu, (xi,yi), method='nearest')
    return grid_yu


def gridSampleRU(vertices, facet):
    """
    get samples of r(u) of the given mesh and make a grid form of the sample
    for the convenience to apply discrete spherical harmonics transform
    .. note: the definition of r(u) can be seen in func::descriptorCS
    ---------------------------------------------------------------------------
    output::grid_ru: numpy array (180,360)
    """
    # get enough sample points to cover the sphere
    # S^2 (-180,180) * (-90,90) (degree based)
    # sample process: multiply triangle edge vectors by
    # the random lengths and sum
    # then offset by the origin to generate sample points
    # in space on the triangle
    vertices, facet = uniform_volume(vertices, facet)
    samples, face_index = triangle_face_sample(vertices, facet,
                                               number=(360 * 180 * 3))
    # align
    centroid = get_trimesh_centroid(vertices, facet)
    samples = samples - np.array([centroid] * samples.shape[0])
    # get the cross product of (v2-v0),(v1-v0) of each face
    # corresponding to the sample points
    tri_cross = np.array([np.cross((vertices[facet[k, 2]]
                                    - vertices[facet[k, 0]]),
                                   (vertices[facet[k, 1]]
                                    - vertices[facet[k, 0]]))
                          for k in range(len(facet))])
    sample_cross = tri_cross[face_index]
    # get the norm of each cross product
    sample_corssnorm = np.linalg.norm(sample_cross, axis=1)
    # get y(u) for each sample point (without normalization)
    yu = np.array([np.dot(samples[k], sample_cross[k])
                   for k in range(len(samples))])
    # normalize
    yu = np.divide(yu, sample_corssnorm)
    # make sure y(u) is non-negative value(since we suppose dot(u,n(u))
    # should non-negtive)
    yu = np.abs(yu)
    # convert the cartesian coordinate to spherical coordinate (and the first
    # coordinate radius is the desired x(u))
    spsam = xyz2sp(samples)
    # compute r(u) for each sample points
    ru = spsam[:, 0] + 1j * yu
    # use nearest point on each grid position as the value and generate the
    # grid form of the sample value r(u)
    xi = np.linspace(-np.pi, np.pi, 360)
    yi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 180)
    grid_ru = griddata(spsam[:, 1:3], ru, (xi[None, :], yi[:, None]),
                       method='nearest')
    # xi,yi = np.mgrid[-np.pi:np.pi:360j, -0.5*np.pi:0.5*np.pi:180j]
    # grid_ru = griddata(spsam[:,1:3], ru, (xi,yi), method='nearest')
    return grid_ru


def descriptorRS(vertices, facet, coef_num_sqrt=13):
    """
    apply spherical harmonics transform on the real function on the sphere S^2
    and use the truncated coefficients as the descriptor of the given mesh
    refer to:
        Description of 3D-shape using a complex function on the sphere
        D.V. Vranic, D. Saupe, 2002
        (this is the ray-based method mentioned in the paper)
    basic idea:
        x: S^2 --> [0,inf) in R, u |--> max{x>=0| xu in mesh-surface or {0}}
        get enough sample of x(u) and apply spherical harmonics transform on it
    ---------------------------------------------------------------------------
    para::coef_num_sqrt: the square root of desired number of the dimensions
        of the shape of the mesh which is also the number of the truncated
        coefficients
    output:: coeffs_trunc: list with size coef_num_sqrt^2
    ..the desired shape descriptor
    """
    # get the sample value of x(u)
    zi = gridSampleXU(vertices, facet)
    # generate the sherical harmonics coefficients
    coeffs = SHExpandDH(zi, sampling=2)
    coeffs_trunc = [[coeffs[0, k, :(k+1)].tolist(),
                     coeffs[1, k, 1:(k + 1)].tolist()]
                    for k in range(coef_num_sqrt)]
    coeffs_trunc = [var for sublist in coeffs_trunc
                    for subsublist in sublist
                    for var in subsublist]
    coeffs_trunc = np.array(coeffs_trunc)
    return coeffs_trunc


def descriptorSS(vertices, facet, coef_num_sqrt=13):
    """
    apply spherical harmonics transform on the imaginary part of complex
    function on the sphere S^2 and use the truncated coefficients as the
    descriptor of the given mesh
    refer to:
        Description of 3D-shape using a complex function on the sphere
        D.V. Vranic, D. Saupe, 2002
        (this is the shading-based method mentioned in the paper)
    basic idea:
        y: S^2 --> [0,inf) in R, u |-->  0  if x(u) = 0 ; dot(u,n(u)),
        otherwise where n(u) is the normal vector of the mesh at the
        point x(u)*u (the fast intersection point on the surface with ray in
        direction u) get enough sample of y(u) and apply spherical harmonics
        transform on it
    ---------------------------------------------------------------------------
    para::coef_num_sqrt: the square root of desired number of the dimensions of
        the shape of the mesh which is also the number of the truncated
        coefficients
    output:: coeffs_trunc: list with size coef_num_sqrt^2
    ..the desired shape descriptor
    """
    # get the sample value of y(u)
    zi = gridSampleYU(vertices, facet)
    # generate the sherical harmonics coefficients
    coeffs = SHExpandDH(zi, sampling=2)
    coeffs_trunc = [[coeffs[0, k, :(k + 1)].tolist(),
                     coeffs[1, k, 1:(k + 1)].tolist()]
                    for k in range(coef_num_sqrt)]
    coeffs_trunc = [var for sublist in coeffs_trunc
                    for subsublist in sublist
                    for var in subsublist]
    coeffs_trunc = np.array(coeffs_trunc)
    return coeffs_trunc


def descriptorCS(vertices, facet, coef_num_sqrt=13):
    """
    apply spherical harmonics transform on the complex function on the sphere
    S^2 and use the truncated coefficients as the descriptor of the given mesh
    refer to:
        Description of 3D-shape using a complex function on the sphere
        D.V. Vranic, D. Saupe, 2002
        (this is the complex feature vector method mentioned in the paper)
    basic idea:
        r: S^2 --> C, u |--> x + y, with
        x: S^2 --> [0,inf) in R, u |--> max{x>=0| xu in mesh-surface or {0}}
        y: S^2 --> [0,inf) in R, u |-->  0  if x(u) = 0 ; dot(u,n(u)),
        otherwise where n(u) is the normal vector of the mesh at the point
        x(u)*u (the fast intersection point on the surface with ray in
        direction u) get enough sample of r(u) and apply spherical harmonics
        transform (complex form) on it
    ---------------------------------------------------------------------------
    para::coef_num_sqrt: the square root of desired number of the dimensions of
        the shape of the mesh which is also the number of the truncated
        coefficients
    output:: coeffs_trunc: list with size coef_num_sqrt^2 (use the absolute
        value)..the desired shape descriptor
    """
    # get the sample value of r(u)
    zi = gridSampleRU(vertices, facet)
    # generate the sherical harmonics coefficients
    coeffs = np.abs(SHExpandDHC(zi, sampling=2))
    coeffs_trunc = [[coeffs[0, k, :(k + 1)].tolist(),
                     coeffs[1, k, 1:(k + 1)].tolist()]
                    for k in range(coef_num_sqrt)]
    coeffs_trunc = [var for sublist in coeffs_trunc
                    for subsublist in sublist
                    for var in subsublist]
    coeffs_trunc = np.array(coeffs_trunc)
    return coeffs_trunc
