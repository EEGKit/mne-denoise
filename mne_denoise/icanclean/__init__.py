"""ICanClean: Independent Component-based Artifact Cleaning.

This module implements reference-based artifact removal using Canonical
Correlation Analysis (CCA) between EEG/MEG channels and reference noise
sensors (e.g., EOG, EMG, accelerometers, or a second electrode layer).

ICanClean is particularly useful for:
- Mobile EEG / MoBI recordings
- High-motion experiments
- Studies where reference sensors are available

References
----------
.. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.

.. [2] Hotelling, H. (1936). Relations between two sets of variates.
       Biometrika, 28(3/4), 321-377.
"""

from .core import ICanClean

__all__ = [
    "ICanClean",
]
