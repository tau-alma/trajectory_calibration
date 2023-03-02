#!/usr/bin/env python
"""Contains helper functions for plotting
"""

import numpy as np
import matplotlib.pyplot as plt


def set_axes_equal(ax=None):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ``ax.set_aspect('equal')`` and ``ax.axis('equal')`` not working for 3D.

    Parameters
    ----------
    ax : Axes3D, optional
        A mplot3d axes. Default ``plt.gca()``.
    """
    if ax is None:
        ax = plt.gca()

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    """Sets the axes radius.

    Parameters
    ----------
    ax : Axes3D
        The axes to set the radius on
    origin : ndarray, shape (3,)
        The desired origin
    radius : ndarray, shape (3,)
        The desired radius
    """
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def plot_axes(T, scale=0.1, ax=None, label=None):
    """Plot the coordinate system given by transformation T

    Parameters
    ----------
    T : ndarray
        4x4 transformation matrix
    scale : float, optional
        The scale of the unit vectors to plot
    ax : Axes3D, optional
        A mplot3d axes to plot on. Default ``plt.gca()``.
    """
    if ax is None:
        ax = plt.gca()

    x = np.array([scale, 0, 0, 1])
    y = np.array([0, scale, 0, 1])
    z = np.array([0, 0, scale, 1])

    xc = np.dot(T, x)
    yc = np.dot(T, y)
    zc = np.dot(T, z)

    ax.plot3D([T[0, -1], xc[0]], [T[1, -1], xc[1]], [T[2, -1], xc[2]], 'r-')
    ax.plot3D([T[0, -1], yc[0]], [T[1, -1], yc[1]], [T[2, -1], yc[2]], 'g-')
    ax.plot3D([T[0, -1], zc[0]], [T[1, -1], zc[1]], [T[2, -1], zc[2]], 'b-')

    if label:
        ax.text(T[0, -1], T[1, -1], T[2, -1], label)


def plot_trajectory(T, *args, ax=None, **kwargs):
    """Plot the trajectory

    Parameters
    ----------
    T : Trajectory
        The trajectory to plot
    ax : Axes3D, optional
        A mplot3d axes to plot on. Default ``plt.gca()``
    args, kwargs
        Passed to matplotlib's plot command
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(*list(T.get_xyz().T), *args, **kwargs)
