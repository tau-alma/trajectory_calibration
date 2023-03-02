#!/usr/bin/env python
"""Contains a single class to represent trajectories
"""

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.utils.tools import (
    as_transition_matrix,
    interpolate_transition
)


class Trajectory(object):
    """A class to represent trajectories

    Attributes
    ----------
    p : list
        (x,y,z) positions of the trajectory poses
    R : list
        3x3 rotation matrices of the trajectory poses
    t : list
        Timestamps for the trajectory poses
    """
    __array_ufunc__ = None  # Tell NumPy to leave the class alone

    def __init__(self, trajectory=[], time=[]):
        """Initialize the trajectory

        Parameters
        ----------
        trajectory : list, optional
            Trajectory poses in homogeneous 4x4 matrix format
        time : list, optional
            Timestamps for trajectory poses

        Raises
        ------
        ValueError
            Raised if the `trajectory` and `time` lengths don't match
        """
        super(Trajectory, self).__init__()
        self.R = []
        self.p = []
        self.t = []

        # Cover the case when only a single point is passed as input
        trajectory = np.reshape(trajectory, (-1, 4, 4))
        time = np.atleast_1d(time)

        # Make sure there's as many points as there are time instances
        if len(trajectory) != len(time):
            raise ValueError(f'Trajectory length {len(trajectory)} does not ' +
                             f'match time length {len(time)}')

        for T, t in zip(trajectory, time):
            self.R.append(T[:3, :3])
            self.p.append(T[:3, -1].reshape(-1, 1))
            self.t.append(t)

    def __len__(self):
        """Return length of trajectory
        """
        return len(self.R)

    def __getitem__(self, key):
        """Enable (multi)indexing using int or slice

        Parameters
        ----------
        key : int or slice
            The (multi)indexing key for the subtrajectory

        Returns
        -------
        Trajectory
            The desired subtrajectory

        Raises
        ------
        ValueError
            Raised if `key` is not int or slice

        Todo
        ----
        Implement boolean indexing
        """
        if isinstance(key, (int, np.integer, slice)):
            return self.__class__(self.as_transitions()[key], self.t[key])
        trajectory = []
        time = []
        transitions = self.as_transitions()
        for k in key:
            if isinstance(k, (int, np.integer)):
                trajectory.append(transitions[k])
                time.append(self.t[k])
            elif isinstance(k, slice):
                trajectory.extend(transitions[k])
                time.extend(self.t[k])
            else:
                raise ValueError(
                    'Only supports (multi)indexing using int or slice.'
                )
        return self.__class__(trajectory, time)

    def __add__(self, X):
        """Combine two trajectories

        Parameters
        ----------
        X : Trajectory
            The trajectory to add to the end

        Returns
        -------
        Trajectory
            The combined trajectory

        Raises
        ------
        ValueError
            Raised if timestamps are not in chronological order
        """
        try:
            if not self.t[-1] < X.t[0]:
                raise ValueError("The timestamps are not chronological.")
        except IndexError:
            pass  # One of the trajectories is empty, which is fine

        return self.__class__(self.as_transitions() + X.as_transitions(),
                              self.t + X.t)

    def __matmul__(self, X):
        """Matrix multiplication to transform entire trajectory

        Parameters
        ----------
        X : ndarray
            Size (4,4) ndarray to multiply the trajectory with

        Returns
        -------
        Trajectory
            The transformed trajectory T @ X
        """
        T = self.as_array() @ X
        return self.__class__(list(T), self.t)

    def __rmatmul__(self, X):
        """Matrix multiplication to transform entire trajectory

        Parameters
        ----------
        X : ndarray
            Size (4,4) ndarray to multiply the trajectory with

        Returns
        -------
        Trajectory
            The transformed trajectory X @ T
        """
        T = X @ self.as_array()
        return self.__class__(list(T), self.t)

    def get_xyz(self):
        """Get xyz values

        Returns
        -------
        ndarray
            Size (n,3) array of xyz values
        """
        return np.squeeze(np.asarray(self.p))

    def get_euler(self):
        """Get rpy values

        Returns
        -------
        ndarray
            Size (n,3) array of rpy values
        """
        return Rotation.from_matrix(self.R).as_euler('ZYX')[:, [2, 1, 0]]

    def get_quaternion(self):
        """Get quaternions

        Returns
        -------
        ndarray
            Size (n,4) array of quaternions in scalar-last-format (qx qy qz qw)
        """
        return Rotation.from_matrix(self.R).as_quat()

    def as_transitions(self):
        """Get homogeneous transition matrices

        Returns
        -------
        list
            Length n list of size (4,4) ndarrays of transition matrices
        """
        return [as_transition_matrix(
            self.R[k], self.p[k]) for k in range(len(self))]

    def as_array(self):
        """Get homogeneous transition matrices

        Returns
        -------
        ndarray
            Size (n,4,4) array of transitions
        """
        return np.asanyarray(self.as_transitions())

    def save(self, file):
        """Save the trajectory in TUM format
        (https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)

        Parameters
        ----------
        file : path-like
            The output file. Existing file will be overwritten.
        """
        with open(file, 'w') as f:
            f.write('#timestamp tx ty tz qx qy qz qw\n')
            for t, p, q in zip(self.t, self.get_xyz(), self.get_quaternion()):
                f.write(
                    str(t) + " " +
                    " ".join(str(n) for n in p) + " " +
                    " ".join(str(n) for n in q) + "\n"
                )

    def relative_pose(self, method="B", n=1):
        """Return trajectory of relative poses w.r.t `method`


        Parameters
        ----------
        method : str, optional
            The method for selecting the reference frame for relative poses.
            Can take the following values:

            - `A`, the first pose
            - `B`, pose k-n
            - `C`, keypoint every n-th point
        n : int, optional
            See meaning from `method`. Ignored for method `A`

        Returns
        -------
        Trajectory
            The relative trajectory

        Raises
        ------
        ValueError
            Raised if `method` not in {`A`, `B`, `C`}
        """
        Tr = self.__class__()
        if method == "A":
            T0inv = as_transition_matrix(self.R[0], self.p[0], inv=True)
            for k in range(1, len(self)):
                rel = T0inv @ as_transition_matrix(self.R[k], self.p[k])
                Tr.R.append(rel[:3, :3])
                Tr.p.append(rel[:3, -1].reshape(-1, 1))
                Tr.t.append(self.t[k])
        elif method == "B":
            for k in range(len(self) - n):
                T0inv = as_transition_matrix(self.R[k], self.p[k], inv=True)
                rel = T0inv @ as_transition_matrix(self.R[k+n], self.p[k+n])
                Tr.R.append(rel[:3, :3])
                Tr.p.append(rel[:3, -1].reshape(-1, 1))
                Tr.t.append(self.t[k+n])
        elif method == "C":
            for k in range(0, len(self)-n, n):
                T0inv = as_transition_matrix(self.R[k], self.p[k], inv=True)
                for r in range(1, n):
                    rel = T0inv @ as_transition_matrix(self.R[k+r], self.p[k+r])
                    Tr.R.append(rel[:3, :3])
                    Tr.p.append(rel[:3, -1].reshape(-1, 1))
                    Tr.t.append(self.t[k+r])
        else:
            raise ValueError("Unknown method.")
        return Tr

    def interpolate(self, t, timeshift=0):
        """Return trajectory interpolated to given timestamps

        Parameters
        ----------
        t : list
            Timestamps to interpolate to
        timeshift : int, optional
            Optional timeshift to apply before interpolation

        Returns
        -------
        Trajectory
            The interpolated trajectory
        """
        t0 = np.asanyarray(self.t) + timeshift
        T0 = self.as_transitions()
        t = np.asanyarray(t)
        T = [Tn for Tn in interpolate_transition(t0, T0, t)]
        return self.__class__(T, t)
