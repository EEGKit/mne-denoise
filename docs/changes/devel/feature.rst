Add publication-quality visualization module (``mne_denoise.viz``) with DSS summary plots, benchmark comparison figures, ZapLine analytics, and a shared colorblind-safe theme.

Add ERP benchmark plot functions (``mne_denoise.viz.erp``): ``plot_erp_signal_diagnostics``, ``plot_erp_condition_interaction``, ``plot_erp_metric_violins``, ``plot_erp_endpoint_summary``, ``plot_erp_pipeline_slopes``, ``plot_erp_grand_average``, ``plot_erp_grand_condition_interaction``, ``plot_erp_null_distribution``, and ``plot_erp_forest``.

Add deferred-group I/O helpers (``mne_denoise.viz.erp_io``): ``ERPGroupData``, ``save_subject_result``, ``load_subject_result``, and ``aggregate_group_results`` for scalable multi-subject ERP benchmark workflows.

Add deferred-group I/O helpers for line-noise benchmarks (``mne_denoise.viz.benchmark_io``): ``LineNoiseGroupData``, ``save_subject_benchmark_results``, ``load_subject_benchmark_results``, and ``aggregate_benchmark_results``.

Rename ZapLine-specific functions at source to avoid name collisions: ``zapline.plot_psd_comparison`` → ``plot_zapline_psd_comparison``, ``zapline.plot_spatial_patterns`` → ``plot_zapline_patterns``.

Migrate ``benchmark.py`` from private ``_resolve_save`` to shared ``_finalize_fig`` for consistent figure saving with ``mkdir -p``.

Extract shared spectral QA metric helpers into ``mne_denoise.viz._metrics``.

Fix ``_finalize_fig`` in ``mne_denoise.viz._theme`` to create parent directories before saving figures.

Add comprehensive test suite for ERP visualization (92 tests, 96 % coverage on ``erp.py``, 97 % on ``erp_io.py``).

Fix ``_try_import_seaborn`` to use ``raise ... from err`` (B904).
