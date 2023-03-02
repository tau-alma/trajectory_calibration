Usage
=====

To calibrate the extrinsics between two sensor trajectories given in `TUM format`_

.. code-block:: python

   from calibration.calibration import calibrate
   from calibration.utils.io import get_trajectories

   trajectories = get_trajectories(['traj0.txt', 'traj1.txt'])
   extrinsics = calibrate(*trajectories)

For further details see :doc:`api`

.. _TUM format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
