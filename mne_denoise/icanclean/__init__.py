"""ICanClean.

This module contains:
- ``compute_icanclean``: The core array-based iCanClean algorithm.
- ``ICanClean``: The Scikit-learn estimator compatible with MNE-Python
  objects or NumPy arrays.

iCanClean removes artifact subspaces shared by primary channels and
reference channels using canonical correlation analysis (CCA) [1]_ [2]_ [3]_.

References
----------
.. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.
.. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
       Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
       23(19), 8214. https://doi.org/10.3390/s23198214
.. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
       Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
       https://doi.org/10.3390/s23020928
"""

from .core import ICanClean, compute_icanclean

__all__ = [
    "ICanClean",
    "compute_icanclean",
]
