#!/usr/bin/env python
"""Contains implementations of multiple calibration algorithms and a unified
method :meth:`calibrate` to run the calibration using the desired algorithm.
"""

import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation
import scipy
import warnings

from calibration.utils.tools import (
    as_rotation_vector,
    as_transition_matrix,
    skew
)


def calibrate(T1, T2, method="B", n=1, algorithm="DNL", **kwargs):
    """Run the calibration using the method defined as argument

    Parameters
    ----------
    T1 : Trajectory
        First sensor trajectory
    T2 : Trajectory
        Second sensor trajectory
    method : str, optional
        The keyframe selection method, one of {`A`, `B`, `C`}. See
        :meth:`calibration.trajectory.Trajectory.relative_pose` for details.
    n : int, optional
        Keyframe selection parameter, see
        :meth:`calibration.trajectory.Trajectory.relative_pose`
        for details.
    algorithm : str, optional
        The calibration algorithm to use. One of

        - `DNL` - Our direct nonlinear optimization
        - `DNLO` - Our direct nonlinear optimization with ourlier rejection
        - `Ali` - Nonlinear optimization with cost Xc from [Ali2019]_
        - `Park` - Separable solution from [Park2020]_
        - `Taylor` - Separable solution from [Taylor2015]_
        - `Zhuang` - Nonlinear solution from [Zhuang1994]_
    **kwargs
        Additional keyword arguments passed to `DNLO` or `Zhuang`

    Returns
    -------
    ndarray, shape (4,4)
        The homogeneous extrinsic sensor to sensor calibration

    Raises
    ------
    ValueError
        Raised if unknown calibration algorithm


    .. [Ali2019] I. Ali, O. Suominen, A. Gotchev, and E. R. Morales,
       "Methods for simultaneous robot-world-hand-eye calibration:
       A comparative study", Sensors, vol. 19, no. 12, p. 2837, 2019.
    .. [Park2020] C. Park, P. Moghadam, S. Kim, S. Sridharan, and C. Fookes,
       "Spatiotemporal camera-lidar calibration: A targetless and structureless
       approach", IEEE Robotics and Automation Letters, vol. 5, no. 2,
       pp. 1556–1563, 2020.
    .. [Taylor2015] Z. Taylor and J. Nieto, "Motion-based calibration of
       multimodal sensor arrays", in 2015 IEEE International Conference on
       Robotics and Automation (ICRA), 2015, pp. 4843–4850.
    .. [Zhuang1994] H. Zhuang and Z. Qu, "A new identification Jacobian for
       robotic hand/eye calibration", IEEE Transactions on Systems, Man, and
       Cybernetics, vol. 24, no. 8, pp. 1284–1287, 1994.
    """
    assert(T1.t == T2.t)
    T1 = T1.relative_pose(method, n)
    T2 = T2.relative_pose(method, n)

    if algorithm == "DNL":
        T = _DNL(T1, T2)
    elif algorithm == "DNLO":
        T = _DNLO(T1, T2, **kwargs)
    elif algorithm == "Ali":
        T = _Ali(T1, T2)
    elif algorithm == "Park":
        T = _Park(T1, T2)
    elif algorithm == "Taylor":
        T = _Taylor(T1, T2)
    elif algorithm == "Zhuang":
        T = _Zhuang(T1, T2, **kwargs)
    else:
        raise ValueError("Unknown algorithm.")
    return T


def _DNL(T1, T2):
    nlp = dict()

    rotvec = ca.SX.sym('rotvec', 3, 1)
    p = ca.SX.sym('p', 3, 1)
    nlp['x'] = ca.vertcat(rotvec, p)
    x0 = np.ones(6)

    nlp['f'] = _DNL_cost(T1, T2, rotvec, p)

    S = ca.nlpsol('S', 'ipopt', nlp, {"ipopt.print_level": 0, "print_time": 0})
    r = S(x0=x0)

    x = np.array(r['x'].full()).flatten()
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    t = x[3:].reshape(-1, 1)

    return as_transition_matrix(R.T, t)  # TODO: why transpose?


def _DNL_cost(T1, T2, rotvec, p):
    R = _axis_angle_to_matrix(rotvec)
    T = ca.vertcat(ca.horzcat(R, p), ca.SX([[0, 0, 0, 1]]))

    cost = 0
    for k in range(len(T1.R)):
        err = (as_transition_matrix(T1.R[k], T1.p[k]) @ T -
               T @ as_transition_matrix(T2.R[k], T2.p[k]))
        cost += ca.sumsqr(err[:3, :])
    return cost


def _DNLO(T1, T2, threshold=0.01):
    nlp = dict()

    rotvec = ca.SX.sym('rotvec', 3, 1)
    p = ca.SX.sym('p', 3, 1)
    s = ca.SX.sym('s', len(T1), 1)  # for outlier detection
    nlp['g'] = ca.sum1(s)
    lbg = 0.5 * len(T1)  # atleast half the points
    ubg = np.inf
    nlp['x'] = ca.vertcat(rotvec, p, s)
    lbx = ca.vertcat(np.full((6, 1), -np.inf), np.zeros((len(T1), 1)))
    ubx = ca.vertcat(np.full((6, 1), np.inf), np.ones((len(T1), 1)))
    x0 = np.ones((6 + len(T1), 1))

    nlp['f'] = _DNLO_cost(T1, T2, rotvec, p, s, threshold)

    S = ca.nlpsol('S', 'ipopt', nlp, {"ipopt.print_level": 0, "print_time": 0})
    r = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    x = np.array(r['x'].full()).flatten()
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    t = x[3:6].reshape(-1, 1)

    return as_transition_matrix(R.T, t)  # TODO: why transpose?


def _DNLO_cost(T1, T2, rotvec, p, s, threshold):
    R = _axis_angle_to_matrix(rotvec)
    T = ca.vertcat(ca.horzcat(R, p), ca.SX([[0, 0, 0, 1]]))

    cost = 0
    for k in range(len(T1.R)):
        err = (as_transition_matrix(T1.R[k], T1.p[k]) @ T -
               T @ as_transition_matrix(T2.R[k], T2.p[k]))
        cost += s[k] * ca.sumsqr(err[:3, :]) - s[k] * threshold
    return cost


def _axis_angle_to_matrix(rotvec):
    """Coversion between axis-angle and rotation matrix
    https://en.wikipedia.org/wiki/Axis-angle_representation#Rotation_vector
    """
    angle = ca.norm_2(rotvec)
    axis = rotvec / angle
    K = ca.cross(ca.repmat(axis, 1, 3), np.eye(3) * -1)

    return np.eye(3) + ca.sin(angle)*K + (1 - ca.cos(angle))*ca.mpower(K, 2)


def _Ali(T1, T2):
    nlp = dict()

    quat = ca.SX.sym('quat', 4, 1)
    p = ca.SX.sym('p', 3, 1)
    nlp['x'] = ca.vertcat(quat, p)
    x0 = np.zeros(7)
    x0[3] = 1

    nlp['f'] = _Ali_cost(T1, T2, quat, p)

    S = ca.nlpsol('S', 'ipopt', nlp, {"ipopt.print_level": 0, "print_time": 0})
    r = S(x0=x0)

    x = np.array(r['x'].full()).flatten()
    R = Rotation.from_quat(x[:4]).as_matrix()
    t = x[4:].reshape(-1, 1)

    return as_transition_matrix(R, t)


def _Ali_cost(T1, T2, quat, p):
    """ Cost Xc from Ali 2019
    """
    R = _quat_to_matrix(quat)
    T = ca.vertcat(ca.horzcat(R, p), ca.SX([[0, 0, 0, 1]]))

    cost = 0
    for k in range(len(T1.R)):
        err = (as_transition_matrix(T1.R[k], T1.p[k]) @ T -
               T @ as_transition_matrix(T2.R[k], T2.p[k]))
        cost += 1 + ca.sumsqr(err[:3, -1])  # norm of unit quaternion is 1
    return cost


def _quat_to_matrix(quat):
    """Coversion between quaternion and rotation matrix
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    """
    s = ca.sqrt(ca.norm_2(quat))

    R = ca.SX.sym('R', 3, 3)
    R[0, 0] = 1 - 2*s*(quat[1]**2 + quat[2]**2)
    R[0, 1] = 2*s*(quat[0]*quat[1] - quat[2]*quat[3])
    R[0, 2] = 2*s*(quat[0]*quat[2] + quat[1]*quat[3])
    R[1, 0] = 2*s*(quat[0]*quat[1] + quat[2]*quat[3])
    R[1, 1] = 1 - 2*s*(quat[0]**2 + quat[2]**2)
    R[1, 2] = 2*s*(quat[1]*quat[2] - quat[0]*quat[3])
    R[2, 0] = 2*s*(quat[0]*quat[2] - quat[1]*quat[3])
    R[2, 1] = 2*s*(quat[1]*quat[2] + quat[0]*quat[3])
    R[2, 2] = 1 - 2*s*(quat[0]**2 + quat[1]**2)

    return R


def _Park(T1, T2):
    """Rotation by aligning rotation vectors (Park et al. 2020)
    """
    numPoints = len(T1.R)

    P = np.zeros((3, numPoints))
    Q = np.zeros((3, numPoints))

    for k in range(numPoints):
        P[:, k] = as_rotation_vector(T2.R[k])
        Q[:, k] = as_rotation_vector(T1.R[k])

    # Compute covariance matrix
    M = P @ Q.T

    # Compute optimal rotation
    #   R = (M'M)^(1/2) M^(-1)
    # solve using singular value decomposition
    U, S, V = np.linalg.svd(M)
    d = np.sign(np.linalg.det(V.T @ U.T))  # ensure right-handed coordinates
    R = V.T @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]]) @ U.T

    return _transition_using_R(T1, T2, R)


def _Taylor(T1, T2):
    """Rotation by Kabsch algorithm (Taylor and Nieto 2015)
    """
    v1 = Rotation.from_matrix(T1.R).as_rotvec()
    v2 = Rotation.from_matrix(T2.R).as_rotvec()
    R = Rotation.align_vectors(v1, v2)[0].as_matrix()
    return _transition_using_R(T1, T2, R)


def _transition_using_R(T1, T2, R):
    """Solve translation part in transition using an existing rotation
    """
    numPoints = len(T1.R)

    # Solve translation and scale using least-squares
    A = np.zeros((numPoints * 3, 3))
    b = np.zeros(numPoints * 3)

    for k in range(numPoints):
        A[k*3:k*3 + 3, :] = np.eye(3) - T1.R[k]
        b[k*3:k*3 + 3] = np.squeeze(T1.p[k] - R @ T2.p[k])

    t = scipy.linalg.lstsq(A, b)[0].reshape(-1, 1)

    return as_transition_matrix(R, t)


def _Zhuang(T1, T2, initial_guess, rtol=1e-15, max_iterations=500, verbose=False):
    rho = initial_guess
    converged = False
    i = 0

    while not converged:
        if i >= max_iterations:
            print('Did not converge')
            break
        i += 1
        drho = np.linalg.pinv(_makeJ(T1, T2, rho)) @ _makeh(T1, T2, rho)
        rho = rho + drho
        if verbose:
            print(i)

        if np.allclose(drho, np.zeros_like(drho), rtol=rtol):
            converged = True

    z = rho[:3]
    p = rho[3:].reshape(3, 1)
    R = _rotation_from_z(z)
    T = as_transition_matrix(R, p)

    return T


def _rotation_from_z(z):
    """Eq 14 from Zhuang
    """
    n = np.linalg.norm(z)

    if n:
        k = z / n
        w = 2 * np.arctan2(np.linalg.norm(z, np.inf),
                           np.linalg.norm(k, np.inf))
    else:
        warnings.warn("The norm of z is zero")
        k = np.array([0, 0, 1])
        w = 0

    return Rotation.from_rotvec(w * k).as_matrix()


def _makeh(T1, T2, rho):
    """Construct the residual vector
    """
    fs = []
    gs = []
    for A, B in zip(T1, T2):
        fs.append(_f(A, B, rho))
        gs.append(_g(A, B, rho))
    return -np.vstack((fs, gs)).flatten()


def _makeJ(T1, T2, rho):
    """Construct the identification Jacobian
    """
    dfdzs = []
    dgdzs = []
    dgdps = []
    for A, B in zip(T1, T2):
        dfdzs.append(_dfdz(A, B, rho))
        dgdzs.append(_dgdz(A, B, rho))
        dgdps.append(_dgdp(A, B, rho))

    dfdzs = np.vstack(dfdzs)
    dgdzs = np.vstack(dgdzs)
    dgdps = np.vstack(dgdps)
    return np.block([[dfdzs, np.zeros_like(dfdzs)],
                     [dgdzs, dgdps]])


def _f(A, B, rho):
    """Eq 14a from Zhuang
    """
    z = rho[:3]

    kA = _get_k(A)
    kB = _get_k(B)

    return skew(kA + kB) @ z - kB + kA


def _get_k(X):
    """Get rotation axis
    """
    rX = Rotation.from_matrix(X.R).as_rotvec().squeeze()
    w = np.linalg.norm(rX)
    return rX / w if w else np.array([0, 0, 1])


def _g(A, B, rho):
    """Eq 14b from Zhuang
    """
    z = rho[:3]
    pX = rho[3:]
    pB = B.p[0].flatten()
    pA = A.p[0].flatten()
    RA = A.R[0]

    u = (RA - np.eye(3)) @ pX + pA
    return skew(u + pB) @ z - pB + u


def _dfdz(A, B, rho):
    """Eq 16a from Zhuang
    """
    kA = _get_k(A)
    kB = _get_k(B)

    return skew(kA + kB)


def _dgdz(A, B, rho):
    """Eq 16b from Zhuang
    """
    pX = rho[3:]
    pB = B.p[0].flatten()
    pA = A.p[0].flatten()
    RA = A.R[0]
    return skew(pB + pA + (RA - np.eye(3)) @ pX)


def _dgdp(A, B, rho):
    """Eq 16c from Zhuang
    """
    z = rho[:3]
    RA = A.R[0]
    return (np.eye(3) - skew(z)) @ (RA - np.eye(3))
