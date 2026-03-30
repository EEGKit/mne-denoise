ZapLine Examples
================

Overview
--------

Examples demonstrating ZapLine and ZapLine-plus for removing power-line
artifacts from synthetic, epoched, continuous, and adaptive-cleaning scenarios.

Files
-----

- ``plot_01_basic_usage.py``: Basic ZapLine usage on synthetic line-noise data.
- ``plot_02_parameter_tuning.py``: Parameter tuning and real NoiseTools MEG data.
- ``plot_03_epoched_data.py``: Epoched ZapLine workflows and high-channel MEG data.
- ``plot_04_adaptive_mode.py``: ZapLine-plus style adaptive cleaning on non-stationary data.
- ``plot_05_adaptive_advanced.py``: Advanced harmonic and chunk-level adaptive outputs.

Data Requirements
-----------------

- Synthetic sections run directly with no external data.
- Examples using MNE datasets download and cache them through MNE when needed.
- NoiseTools-backed examples download and cache the required `.mat` files into
  ``examples/zapline/data`` the first time they are run.

References
----------

- de Cheveigné (2020). ZapLine: A simple and effective method to remove power line artifacts. NeuroImage.
- Klug & Kloosterman (2022). Zapline-plus: A Zapline extension for automatic and adaptive removal of frequency-specific noise artifacts in M/EEG. Human Brain Mapping.
