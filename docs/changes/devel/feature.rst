Restructure ``mne_denoise.viz`` into content-based public modules: ``theme``,
``components``, ``signals``, ``spectra``, ``stats``, and ``summary``.

Consolidate method-specific helper plots into shared generic component,
signal, spectral, and grouped-metric plotting functions, and move composed
multi-panel dashboards into ``mne_denoise.viz.summary``.

Add ERP summary plots to ``mne_denoise.viz``: ``plot_erp_signal_diagnostics``,
``plot_erp_condition_interaction``, ``plot_erp_endpoint_summary``, and
``plot_erp_grand_condition_interaction``. Generic group-statistic plots are
exposed as ``plot_metric_violins``, ``plot_metric_slopes``,
``plot_null_distribution``, and ``plot_forest``; grouped evoked overlays are
exposed as ``plot_grand_average_evokeds``.

Fix ``_finalize_fig`` in ``mne_denoise.viz.theme`` to create parent
directories before saving figures, and fix ``_try_import_seaborn`` to use
``raise ... from err`` (B904).
