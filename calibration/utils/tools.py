#!/usr/bin/env python
"""Contains helper functions for general tasks
"""

import numpy as np
import scipy
from scipy.spatial.transform import Rotation


def as_transition_matrix(R, t, inv=False):
    """Returns a transition matrix

    Parameters
    ----------
    R : ndarray
        Rotation matrix
    t : ndarray
        Translation vector
    inv : bool
        Invert transition

    Returns
    -------
    ndarray
        Transition matrix
    """
    if inv:
        return np.vstack([np.hstack([R.T, -R.T @ t]), [0, 0, 0, 1]])
    else:
        return np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])


def interpolate_transition(t0, T0, t):
    """Interpolate between transitions

    Parameters
    ----------
    t0 : ndarray
        Time instances for transitions
    T0 : list of ndarray
        Transition matrices
    t : ndarray
        Times to compute interpolations at

    Yields
    ------
    ndarray
        The interpolated transitions
    """
    hi = np.minimum(len(t0) - 1, np.searchsorted(t0, t, 'right'))
    lo = np.maximum(0, hi - 1)
    with np.errstate(divide='ignore'):
        interpolation_ratio = (t - t0[lo]) / (t0[hi] - t0[lo])

    for i, j, a in zip(lo, hi, interpolation_ratio):
        if np.isinf(a):
            yield np.full((4, 4), np.nan)
        else:
            relative_pose = scipy.linalg.logm(np.linalg.inv(T0[i]) @ T0[j])
            yield T0[i] @ scipy.linalg.expm(a * relative_pose)


def as_rotation_vector(R):
    """Transform rotation matrices to rotation vectors

    Parameters
    ----------
    R : array_like, shape (n,3,3) or (3,3)
        A single rotation matrix or a stack of matrices, where `matrix[i]`
        is the i-th matrix.

    Returns
    -------
    ndarray, shape (3,) or (n,3)
        The rotation vectors corresponding to the inputs
    """
    return Rotation.from_matrix(R).as_rotvec()


def wrap_to_pi(angle):
    """Wrap the input angle between [-pi,pi)

    Parameters
    ----------
    angle : double
        Angle in radians

    Returns
    -------
    double
        The input wrapped between [-pi,pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def as_axis_angle(R):
    """Transform rotation matrices to axis-angle

    Parameters
    ----------
    R : array_like, shape (3,3)
        A single rotation matrix

    Returns
    -------
    axis : ndarray, shape (3,)
        The rotation axis
    angle : double
        The rotation angle
    """
    rotvec = as_rotation_vector(R)
    theta = wrap_to_pi(np.linalg.norm(rotvec))
    return rotvec / theta if theta else rotvec, theta


def skew(v):
    """Returns a skew symmetric matrix corresponding to v

    Parameters
    ----------
    v : arraylike
        Vector of size (3,) where v=[vx, vy, vz]

    Returns
    -------
    ndarray
        The skew symmetric matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rotational_error(T1, T2, T):
    """Calculates the mean relative rotation error

    Parameters
    ----------
    T1 : Trajectory
        The first sensor trajectory
    T2 : Trajectory
        The second sensor trajectory
    T : ndarray, shape (4,4)
        Sensor to sensor calibration

    Returns
    -------
    double
        The mean relative rotation error in degrees
    """
    err = []
    for T1i, T2i in zip(T1, T2):
        R = np.linalg.inv(T[:3, :3] @ T2i.R) @ (T1i.R @ T[:3, :3])
        err.append(np.rad2deg(as_axis_angle(R)[1]))
    return np.mean(err)


def translational_error(T1, T2, T):
    """Calculates the mean relative translation error

    Parameters
    ----------
    T1 : Trajectory
        The first sensor trajectory
    T2 : Trajectory
        The second sensor trajectory
    T : ndarray, shape (4,4)
        Sensor to sensor calibration

    Returns
    -------
    double
        The mean relative translation error
    """
    err = []
    for T1i, T2i in zip(T1.as_transitions(), T2.as_transitions()):
        err.append((T1i @ T - T @ T2i)[:3, 3])
    return np.mean(np.linalg.norm(err, axis=1))


def get_relative_error(T1, T2, T):
    """Get relative calibration errors

    Parameters
    ----------
    T1 : Trajectory
        The first sensor trajectory
    T2 : Trajectory
        The second sensor trajectory
    T : ndarray, shape (4,4)
        Sensor to sensor calibration

    Returns
    -------
    translation : double
        The mean relative translation error
    rotation : double
        The mean relative rotation error in degrees
    """
    return translational_error(T1, T2, T), rotational_error(T1, T2, T)


def get_absolute_error(T, GT):
    """Get absolute calibration errors

    Parameters
    ----------
    T : ndarray, shape (4,4)
        Sensor to sensor calibration
    GT : ndarray, shape (4,4)
        Ground truth calibration

    Returns
    -------
    translation : double
        The absolute translation error
    rotation : double
        The absolute rotation error in degrees
    """
    R = np.linalg.inv(T[:3, :3]) @ GT[:3, :3]
    return (np.linalg.norm(GT[:3, 3] - T[:3, 3]),
            np.rad2deg(as_axis_angle(R)[1]))
