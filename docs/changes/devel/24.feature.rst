**Viz Major Refactor**:

- Restructured ``mne_denoise.viz`` into content-based modules:
  ``theme``, ``components``, ``signals``, ``spectra``, ``stats``, and
  ``summary``.
- Removed method-scoped plotting modules and exports
  (``comparison``, ``dss``, ``zapline``) from the public Viz surface.
- Standardized the Viz API around explicit, study-agnostic plotting inputs.

**QA Module Addition**:

- Added ``mne_denoise.qa`` as the canonical quality-assurance module.
- Added low-level array-based spectral QA metrics (artifact suppression and
  signal-preservation metrics).
- Added high-level Raw-based convenience helpers for PSD aggregation and
  benchmark metric computation.

**What Each Viz Module Does**:

- ``mne_denoise.viz.theme``: Shared publication-ready style utilities.
  ``set_theme`` and ``use_theme`` apply rc settings; ``get_theme_rc`` returns
  merged rc dicts; ``themed_figure`` and ``themed_legend`` create styled
  figure/legend objects; ``style_axes`` applies axis-level finishing;
  ``get_color`` and ``get_series_color`` provide semantic and series colors.

- ``mne_denoise.viz.components``: Component-level diagnostics.
  ``plot_component_score_curve`` shows score/eigenvalue curves;
  ``plot_component_patterns`` plots spatial patterns/topomaps;
  ``plot_component_summary`` composes score/pattern/signal panels;
  ``plot_component_epochs_image`` renders epoch-by-time component images;
  ``plot_component_time_series`` plots component traces;
  ``plot_component_spectrogram`` visualizes component time-frequency content.

- ``mne_denoise.viz.signals``: Time-domain and evoked signal comparisons.
  ``plot_evoked_gfp_comparison`` compares GFP curves;
  ``plot_channel_time_course_comparison`` compares selected channel traces;
  ``plot_power_ratio_map`` maps after/before channel power ratios;
  ``plot_signal_overlay`` overlays one selected before/after trace;
  ``plot_grand_average_evokeds`` plots grouped grand-average evoked signals.

- ``mne_denoise.viz.spectra``: Frequency and time-frequency views.
  ``plot_psd_comparison`` compares before/after PSD;
  ``plot_psd_zoom_comparison`` adds harmonic-focused zoom panels;
  ``plot_psd_gallery`` shows multi-series PSDs across zoom targets;
  ``plot_psd_overlay`` overlays multiple PSD series and harmonics;
  ``plot_component_psd_comparison`` compares raw PSD to selected components;
  ``plot_spectrogram_comparison`` compares before/after spectrograms;
  ``plot_time_frequency_mask`` visualizes binary/continuous TF masks;
  ``plot_narrowband_score_scan`` plots narrowband score scans across
  frequencies.

- ``mne_denoise.viz.stats``: Grouped metric and distribution plots.
  ``plot_metric_bars`` draws grouped metric bars with uncertainty;
  ``plot_tradeoff_scatter`` draws metric trade-off scatter plots;
  ``plot_metric_comparison`` draws paired subject/group metric comparisons;
  ``plot_metric_slopes`` draws subject-level slope trajectories across groups;
  ``plot_metric_violins`` draws grouped distribution violins;
  ``plot_null_distribution`` plots observed-vs-null distributions;
  ``plot_forest`` draws grouped effect/CI forest plots;
  ``plot_harmonic_attenuation`` plots per-harmonic attenuation summaries.

- ``mne_denoise.viz.summary``: Thin multi-panel composers built from Viz
  primitives.
  ``plot_denoising_summary`` composes map/PSD/GFP diagnostics;
  ``plot_metric_tradeoff_summary`` composes trade-off and paired metric panels;
  ``plot_component_cleaning_summary`` composes component-cleaning dashboards;
  ``plot_signal_diagnostics_summary`` composes signal-level diagnostics;
  ``plot_condition_interaction_summary`` composes condition interaction panels;
  ``plot_group_condition_interaction_summary`` composes grouped condition
  interaction panels;
  ``plot_endpoint_metrics_summary`` composes endpoint metric overview panels.
