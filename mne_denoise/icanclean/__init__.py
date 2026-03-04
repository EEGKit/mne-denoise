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

.. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
       Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
       23(19), 8214. doi:10.3390/s23198214.

.. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
       Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
       doi:10.3390/s23020928.

Licensing Note
--------------
The EEGLAB iCanClean plugin states that the software implements patented
methods (WO2022061322A1). This implementation derives entirely from the
published papers above and uses standard signal-processing operations.
"""

from .core import ICanClean

__all__ = [
    "ICanClean",
]
