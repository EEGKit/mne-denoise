Testing
=======

Running the test suite
----------------------

The test suite uses `pytest <https://docs.pytest.org/>`_ with branch coverage
via ``pytest-cov``. From the repository root:

.. code-block:: bash

   # Run the full suite with a coverage summary
   python -m pytest tests/ --cov=mne_denoise --cov-report=term-missing -q

   # Run a single test file
   python -m pytest tests/test_viz.py -v

   # Run tests matching a keyword
   python -m pytest tests/ -k "zapline" -v

Coverage targets
----------------

The project enforces the following coverage thresholds (configured in
``codecov.yml``):

* **Project-level**: ≥ 80 % (with 1 % tolerance)
* **Patch-level**: auto (all new/changed lines should be tested when feasible)

Branch coverage is enabled in ``pyproject.toml``:

.. code-block:: toml

   [tool.coverage.run]
   branch = true
   source = ["mne_denoise"]

Test organisation
-----------------

All tests live under the ``tests/`` directory:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Scope
   * - ``test_linear_dss.py``
     - Linear DSS estimators, bias functions, and round-trip fitting
   * - ``test_nonlinear_dss.py``
     - Spectral, periodic, temporal, and TF-mask DSS variants
   * - ``test_zapline_core.py``
     - Core ZapLine estimator (standard and adaptive modes)
   * - ``test_zapline_adaptive.py``
     - Adaptive ZapLine–Plus convergence and chunk handling
   * - ``test_viz.py``
     - DSS and ZapLine visualization basics (PSD, scores, patterns, cleaning)
   * - ``test_viz_dss.py``
     - DSS summary dashboards and helper functions
   * - ``test_viz_benchmark.py``
     - Benchmark visualization (9 plot functions + helper utilities)
   * - ``test_viz_zapline_extended.py``
     - Adaptive and standard ZapLine dashboards, 5 private helpers
   * - ``test_viz_metrics.py``
     - Suppression ratio, spectral distortion, variance removed
   * - ``test_benchmark_io.py``
     - Save/load/aggregate benchmark I/O and JSON encoding
   * - ``test_qa_metrics.py``
     - Peak attenuation metric
   * - ``test_utils.py``
     - ``extract_data_from_mne`` / ``reconstruct_mne_object`` round-trips
   * - ``test_averaging.py``
     - ``AverageBias._apply_datasets`` (group-level averaging)
   * - ``test_erp_viz.py``
     - ERP benchmarking visualization helpers

Writing tests
-------------

* Use the ``matplotlib.use("Agg")`` backend (set automatically in
  ``conftest.py``) and always pass ``show=False`` to plotting functions.
* The ``close_plots`` autouse fixture calls ``plt.close("all")`` after each
  test to avoid memory leaks.
* For visualization tests, assert that the return value is a ``plt.Figure``
  and optionally test file saving with ``tmp_path``.
* Mock estimator classes (e.g. ``MockDSS``, ``MockStandardZapLine``) should
  provide the minimum attributes required by the function under test.

Markers and skipping
--------------------

* ``@pytest.mark.skipif`` is used for tests that depend on optional packages
  (e.g. MATLAB parity tests requiring HDF5 reference data).
* ``pytest.ini_options`` in ``pyproject.toml`` sets ``--strict-markers`` and
  ``--strict-config``.
