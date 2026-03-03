Add publication-quality visualization module (``mne_denoise.viz``) with DSS summary plots, benchmark comparison figures, ZapLine analytics, and a shared colorblind-safe theme.

Add ERP benchmark plot functions (``mne_denoise.viz.erp``): ``plot_erp_signal_diagnostics``, ``plot_condition_interaction``, ``plot_metric_violins``, ``plot_endpoint_summary``, ``plot_pipeline_slopes``, ``plot_grand_average_erp``, ``plot_grand_condition_interaction``, ``plot_null_distribution``, and ``plot_forest``.

Add deferred-group I/O helpers (``mne_denoise.viz.erp_io``): ``ERPGroupData``, ``save_subject_result``, ``load_subject_result``, and ``aggregate_group_results`` for scalable multi-subject ERP benchmark workflows.

Fix ``_finalize_fig`` in ``mne_denoise.viz._theme`` to create parent directories before saving figures.

Add comprehensive test suite for ERP visualization (92 tests, 96 % coverage on ``erp.py``, 97 % on ``erp_io.py``).

Fix ``_try_import_seaborn`` to use ``raise ... from err`` (B904).
