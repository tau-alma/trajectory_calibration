# Motion-based extrinsic calibration

This package implements motion-based sensor to sensor extrinsic calibration by solving the hand-eye calibration problem.

If you use the work in an academic context, please cite:

    @article{valimaki2023,
      author = {Välimäki, Tuomas and Garigipati, Bharath and Ghabcheloo, Reza},
      title = {Motion-Based Extrinsic Sensor-to-Sensor Calibration: Effect of Reference Frame Selection for New and Existing Methods},
      journal = {Sensors},
      volume = {23},
      year = {2023},
      number = {7},
      pages = {3740},
      url = {https://www.mdpi.com/1424-8220/23/7/3740},
      issn = {1424-8220},
      doi = {10.3390/s23073740}
    }

## Installation

Clone repository to a desired location and install using `pip`

```bash
git clone git@github.com:tau-alma/trajectory_calibration.git calibration
cd calibration
pip install .
```

## Usage

To calibrate the extrinsics between two sensor trajectories given in [TUM format]

```python
from calibration.calibration import calibrate
from calibration.utils.io import get_trajectories

trajectories = get_trajectories(['traj0.txt', 'traj1.txt'])
extrinsics = calibrate(*trajectories)
```

Refer to documentation for additional details.

## Documentation

The documentation is available through [Read the Docs]

## Contributing

Follow [PEP8] quidelines for code formatting and use [NumPy style docstring]. The use of code checkers is highly recommended.

[TUM format]: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
[PEP8]: https://www.python.org/dev/peps/pep-0008/
[NumPy style docstring]: https://numpydoc.readthedocs.io/en/latest/format.html
[Read the Docs]: http://trajectory-calibration.readthedocs.io
