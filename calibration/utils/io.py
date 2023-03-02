#!/usr/bin/env python
"""Contains helper functions for I/O operations
"""

import csv
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.trajectory import Trajectory
from calibration.utils.tools import as_transition_matrix


def get_trajectories(trajectories, estimate=False):
    """Read trajectories from file in TUM format
    (https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)

    Parameters
    ----------
    trajectories : list
        A list of trajectory files to read
    estimate : bool, optional
        If ``True``, tries to match the timestamps of `trajectories[1:]` with
        `trajectories[0]` based on when the absolute Cartesian movement is
        above a threshold (5e-2)

    Returns
    -------
    list
        List of :class:`calibration.trajectory.Trajectory`

    TODO
    ----
    Make the estimation threshold a parameter
    """
    # Get reference time and transition
    t = []
    T = []
    with open(trajectories[0], newline='') as f:
        rdr = csv.reader((row for row in f if not row.startswith("#")),
                         delimiter=" ")
        for row in rdr:
            t.append(float(row[0]))
            T.append(_row_as_transition_matrix(row))
    t = np.array(t)

    if estimate:
        print("Estimating time difference.")
        threshold = 5e-2
        d = np.linalg.norm(np.cumsum(np.diff(T, axis=0)[::, -1], axis=0), axis=1)
        start = t[np.argmax(d > threshold)]
        print(f"Start time of reference '{trajectories[0]}': {start}")

    ti = []
    Ti = []
    for trajectory in trajectories[1:]:
        # Get time and transitions from trajectory
        t0 = []
        T0 = []
        with open(trajectory, newline='') as f:
            rdr = csv.reader((row for row in f if not row.startswith("#")),
                             delimiter=" ")
            for row in rdr:
                t0.append(float(row[0]))
                T0.append(_row_as_transition_matrix(row))
        t0 = np.array(t0)

        # Estimate time difference
        if estimate:
            d0 = np.linalg.norm(np.cumsum(np.diff(T0, axis=0)[::, -1], axis=0), axis=1)
            start0 = t0[np.argmax(d0 > threshold)]
            time_diff = start - start0
            print(f"Start time of '{trajectory}': {start0}")
            print(f"  Estimated difference: {time_diff}")
            t0 = t0 + time_diff

        ti.append(t0)
        Ti.append(T0)

    return [Trajectory(Ts, ts) for Ts, ts in zip([T] + Ti, [t] + ti)]


def _row_as_transition_matrix(row):
    """Returns a transition matrix from one TUM format row

    Parameters
    ----------
    row : list
        Row of the form 'timestamp tx ty tz qx qy qz qw'

    Returns
    -------
    ndarray
        Transition matrix
    """
    R = Rotation.from_quat(row[4:]).as_matrix()
    t = np.array(row[1:4], dtype=np.double).reshape((3, 1))
    return as_transition_matrix(R, t)
