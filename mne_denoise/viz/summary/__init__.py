"""Composed multi-panel summary figures for mne-denoise visualizations.

Thin facade that re-exports every public name from the domain submodules
(``_benchmark``, ``_dss``, ``_zapline``, ``_erp``) and the shared
constants / helpers in ``_panels``.

Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
         Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

# -- benchmark summaries -----------------------------------------------------
from ._benchmark import (  # noqa: F401
    plot_denoising_summary,
    plot_metric_tradeoff_summary,
)

# -- DSS summaries -----------------------------------------------------------
from ._dss import (  # noqa: F401
    plot_dss_mode_comparison,
    plot_dss_segmented_summary,
    plot_dss_summary,
)

# -- ERP summaries -----------------------------------------------------------
from ._erp import (  # noqa: F401
    plot_erp_condition_interaction,
    plot_erp_endpoint_summary,
    plot_erp_grand_condition_interaction,
    plot_erp_signal_diagnostics,
)

# -- shared constants & helpers (kept importable from summary for compat) ----
from ._panels import (  # noqa: F401
    _ERP_WIN_COLORS,
    DEFAULT_ERP_WINDOWS,
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHOD_ORDER,
    DEFAULT_PIPE_LABELS,
    DEFAULT_PIPE_ORDER,
    _pipe_color,
    _pipe_label,
)

# -- ZapLine summaries -------------------------------------------------------
from ._zapline import (  # noqa: F401
    plot_zapline_adaptive_summary,
    plot_zapline_cleaning_summary,
    plot_zapline_summary,
)
