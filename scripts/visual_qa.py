#!/usr/bin/env python
"""Visual QA — Exercise every mne_denoise.viz function on real data.

Loads the Runabout oddball dataset (sub-01), runs DSS + ZapLine,
then calls every public plot function with ``show=False`` and saves
PNG files to ``data/runabout/derivatives/mne-denoise/visual_qa/``.

Run with:
    python scripts/visual_qa.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mne_denoise import viz
from mne_denoise.dss import DSS, AverageBias
from mne_denoise.zapline import ZapLine

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ── paths ─────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "runabout"
OUT_DIR = DATA_ROOT / "derivatives" / "mne-denoise" / "visual_qa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sub-directories for organized output
COMP_DIR = OUT_DIR / "comparison"
COMP_DIR.mkdir(exist_ok=True)
COMPS_DIR = OUT_DIR / "components"
COMPS_DIR.mkdir(exist_ok=True)
DSS_DIR = OUT_DIR / "dss"
DSS_DIR.mkdir(exist_ok=True)
ZAP_DIR = OUT_DIR / "zapline"
ZAP_DIR.mkdir(exist_ok=True)
BENCH_DIR = OUT_DIR / "benchmark"
BENCH_DIR.mkdir(exist_ok=True)
THEME_DIR = OUT_DIR / "theme"
THEME_DIR.mkdir(exist_ok=True)

# ── config ────────────────────────────────────────────────────────────
SUB = "sub-01"
LINE_FREQ = 50.0
RESAMPLE_FREQ = 250
HP_FREQ = 0.5
LP_FREQ = 40.0
DSS_N_COMPONENTS = 5
DSS_N_KEEP = 3

passed, failed = [], []


def _try(name: str, func, *args, **kwargs):
    """Run func, track pass/fail, close figures."""
    print(f"  {name} ... ", end="", flush=True)
    try:
        result = func(*args, **kwargs)
        passed.append(name)
        print("✓")
        return result
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"FAILED: {exc}")
        return None
    finally:
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════
# 1 ── Load & preprocess one subject
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Visual QA — loading & preprocessing data")
print("=" * 60)

eeg_dir = DATA_ROOT / SUB / "eeg"
vhdr = sorted(eeg_dir.glob("*.vhdr"))[0]
raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)

# Clean channel names
mapping = {}
for ch in raw.ch_names:
    cleaned = ch.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
    if cleaned != ch:
        mapping[ch] = cleaned
if mapping:
    raw.rename_channels(mapping)

raw.resample(RESAMPLE_FREQ, verbose=False)
sfreq = raw.info["sfreq"]
raw.pick_types(eeg=True, verbose=False)

if "FCz" not in raw.ch_names:
    raw.add_reference_channels("FCz")
raw.set_montage("standard_1020", on_missing="warn")

mastoids = [ch for ch in ["TP9", "TP10"] if ch in raw.ch_names]
if len(mastoids) == 2:
    raw.set_eeg_reference(mastoids, verbose=False)
    raw.drop_channels(mastoids)
else:
    raw.set_eeg_reference("average", verbose=False)

raw.filter(l_freq=HP_FREQ, h_freq=None, verbose=False)

# ── ZapLine (before bandpass so we can visualize line noise removal) ──
print("  Running ZapLine ...")
raw_before_zap = raw.copy()
zap = ZapLine(sfreq=sfreq, line_freq=LINE_FREQ, n_remove="auto", n_harmonics=3)
raw = zap.fit_transform(raw)
print(f"  ZapLine removed {getattr(zap, 'n_removed_', '?')} components")

# Low-pass for ERP analysis
raw.filter(l_freq=None, h_freq=LP_FREQ, verbose=False)

# ── Epoch ──
print("  Epoching ...")
trig_dir = DATA_ROOT / "derivatives" / "trigger_corrected" / SUB / "eeg"
evt_file = sorted(trig_dir.glob("*_events.tsv"))[0]
tc_df = pd.read_csv(evt_file, sep="\t")
tc_df = tc_df[tc_df["trial_type"] != "empty"].reset_index(drop=True)
tc_df["sample_rs"] = np.round(tc_df["onset"].values * (sfreq / 500.0)).astype(int)

unique_types = sorted(tc_df["trial_type"].unique())
event_id = {tt: i + 1 for i, tt in enumerate(unique_types)}
events_arr = np.column_stack(
    [
        tc_df["sample_rs"].values,
        np.zeros(len(tc_df), dtype=int),
        np.array([event_id[tt] for tt in tc_df["trial_type"]]),
    ]
)
valid = (events_arr[:, 0] >= 0) & (events_arr[:, 0] < raw.n_times - 1)
events_arr = events_arr[valid]

epochs = mne.Epochs(
    raw,
    events_arr,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),
    preload=True,
    verbose=False,
)
epochs.drop_bad(reject={"eeg": 125e-6}, verbose=False)
print(f"  {len(epochs)} epochs after rejection")

# ── DSS on epochs ──
print("  Fitting DSS (AverageBias) ...")
n_ep = len(epochs)
epochs_train = epochs[np.arange(0, n_ep, 2)]
epochs_test = epochs[np.arange(1, n_ep, 2)]

bias = AverageBias(axis="epochs")
dss = DSS(bias=bias, n_components=DSS_N_COMPONENTS, return_type="sources")
dss.fit(epochs_train)

sources_train = dss.transform(epochs_train)
sources_kept = sources_train.copy()
sources_kept[:, DSS_N_KEEP:, :] = 0
recon = dss.inverse_transform(sources_kept)
if recon.ndim == 2:
    recon = recon[np.newaxis, ...]

epochs_denoised = mne.EpochsArray(
    recon,
    epochs_train.info,
    tmin=epochs_train.tmin,
    verbose=False,
)
print(f"  DSS eigenvalues: {dss.eigenvalues_[:5].round(5)}")

evoked_orig = epochs_train.average()
evoked_denoised = epochs_denoised.average()
info = epochs.info

# Data arrays for zapline viz
data_before_zap = raw_before_zap.get_data()
data_after_zap = raw.get_data()


# ══════════════════════════════════════════════════════════════════════
# 2 ── Theme tests
# ══════════════════════════════════════════════════════════════════════
print("\n── Theme ──")


def test_use_style():
    """Test use_style context manager."""
    with viz.use_style():
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4], label="test")
        ax.set_title("use_style() context manager")
        ax.legend()
        fig.savefig(THEME_DIR / "use_style.png", dpi=150, bbox_inches="tight")
    return fig


_try("use_style", test_use_style)


def test_pub_figure():
    """Test pub_figure helper."""
    fig, ax = viz.pub_figure()
    ax.plot(np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    ax.set_title("pub_figure()")
    fig.savefig(THEME_DIR / "pub_figure.png", dpi=150, bbox_inches="tight")
    return fig


_try("pub_figure", test_pub_figure)


def test_get_color():
    """Verify _get_color returns correct colors for all keys."""
    from mne_denoise.viz import _get_color

    for key in ["blue", "orange", "green", "dss", "zapline", "clean"]:
        c = _get_color(key)
        assert c is not None, f"_get_color({key!r}) returned None"
    return True


_try("_get_color", test_get_color)


# ══════════════════════════════════════════════════════════════════════
# 3 ── Comparison module
# ══════════════════════════════════════════════════════════════════════
print("\n── Comparison ──")

_try(
    "plot_psd_comparison",
    viz.plot_psd_comparison,
    epochs_train,
    epochs_denoised,
    fmax=45,
    show=False,
    fname=str(COMP_DIR / "psd_comparison.png"),
)

_try(
    "plot_evoked_comparison",
    viz.plot_evoked_comparison,
    evoked_orig,
    evoked_denoised,
    show=False,
    fname=str(COMP_DIR / "evoked_comparison.png"),
)

_try(
    "plot_time_course_comparison",
    viz.plot_time_course_comparison,
    epochs_train,
    epochs_denoised,
    picks=["Cz", "Pz"],
    show=False,
    fname=str(COMP_DIR / "time_course_comparison.png"),
)

_try(
    "plot_power_map",
    viz.plot_power_map,
    epochs_train,
    epochs_denoised,
    info=info,
    show=False,
    fname=str(COMP_DIR / "power_map.png"),
)

_try(
    "plot_spectrogram_comparison",
    viz.plot_spectrogram_comparison,
    epochs_train,
    epochs_denoised,
    fmin=1,
    fmax=40,
    show=False,
    fname=str(COMP_DIR / "spectrogram_comparison.png"),
)

_try(
    "plot_denoising_summary",
    viz.plot_denoising_summary,
    epochs_train,
    epochs_denoised,
    info=info,
    show=False,
    fname=str(COMP_DIR / "denoising_summary.png"),
)

_try(
    "plot_overlay_comparison",
    viz.plot_overlay_comparison,
    epochs_train,
    epochs_denoised,
    show=False,
    fname=str(COMP_DIR / "overlay_comparison.png"),
)


# ── spectral PSD comparison (needs component data + sfreq) ──
def _spectral_psd():
    # Use DSS sources as "components"
    sources = dss.transform(epochs_train)  # (n_epochs, n_comp, n_times)
    avg_sources = sources.mean(axis=0)  # (n_comp, n_times)
    return viz.plot_spectral_psd_comparison(
        epochs_train,
        avg_sources,
        sfreq=sfreq,
        fmin=1,
        fmax=40,
        show=False,
        fname=str(COMP_DIR / "spectral_psd_comparison.png"),
    )


_try("plot_spectral_psd_comparison", _spectral_psd)


# ══════════════════════════════════════════════════════════════════════
# 4 ── Components module
# ══════════════════════════════════════════════════════════════════════
print("\n── Components ──")

_try(
    "plot_score_curve",
    viz.plot_score_curve,
    dss,
    show=False,
    fname=str(COMPS_DIR / "score_curve.png"),
)

_try(
    "plot_spatial_patterns [components]",
    viz.components.plot_spatial_patterns,
    dss,
    info=info,
    n_components=5,
    show=False,
    fname=str(COMPS_DIR / "spatial_patterns.png"),
)

_try(
    "plot_component_summary",
    viz.plot_component_summary,
    dss,
    data=epochs_train,
    info=info,
    n_components=3,
    show=False,
    fname=str(COMPS_DIR / "component_summary.png"),
)

_try(
    "plot_component_image",
    viz.plot_component_image,
    dss,
    data=epochs_train,
    n_components=3,
    show=False,
    fname=str(COMPS_DIR / "component_image.png"),
)

_try(
    "plot_component_time_series",
    viz.plot_component_time_series,
    dss,
    data=epochs_train,
    n_components=3,
    show=False,
    fname=str(COMPS_DIR / "component_time_series.png"),
)


def _narrowband_scan():
    freqs = np.arange(1, 41)
    evals = np.random.default_rng(42).exponential(0.1, size=len(freqs))
    evals[9] = 0.8  # spike at 10 Hz
    return viz.plot_narrowband_scan(
        freqs,
        evals,
        peak_freq=10.0,
        show=False,
        fname=str(COMPS_DIR / "narrowband_scan.png"),
    )


_try("plot_narrowband_scan", _narrowband_scan)


def _tf_mask():
    times = np.linspace(-0.2, 0.8, 100)
    freqs = np.arange(1, 41)
    mask = np.zeros((len(freqs), len(times)))
    mask[8:12, 30:70] = 1.0  # 9–12 Hz, 100–500 ms
    return viz.plot_tf_mask(
        mask,
        times,
        freqs,
        title="Example TF Mask",
        show=False,
        fname=str(COMPS_DIR / "tf_mask.png"),
    )


_try("plot_tf_mask", _tf_mask)


def _component_spectrogram():
    sources = dss.transform(epochs_train)
    comp0 = sources[:, 0, :]  # (n_epochs, n_times)
    return viz.plot_component_spectrogram(
        comp0,
        sfreq=sfreq,
        title="DSS Component 1 Spectrogram",
        show=False,
        fname=str(COMPS_DIR / "component_spectrogram.png"),
    )


_try("plot_component_spectrogram", _component_spectrogram)


# ══════════════════════════════════════════════════════════════════════
# 5 ── DSS module
# ══════════════════════════════════════════════════════════════════════
print("\n── DSS ──")

_try("plot_dss_eigenvalues", viz.plot_dss_eigenvalues, dss, show=False)

_try(
    "plot_dss_patterns",
    viz.plot_dss_patterns,
    dss,
    info=info,
    max_components=5,
    show=False,
)

# plot_dss_summary expects 2D (n_channels, n_times) data
_try(
    "plot_dss_summary",
    viz.plot_dss_summary,
    dss,
    data_before=epochs_train.average().get_data(),
    data_after=epochs_denoised.average().get_data(),
    sfreq=sfreq,
    channel_names=epochs_train.ch_names,
    info=info,
    max_components=4,
    title="DSS Summary (sub-01, AverageBias)",
    show=False,
)


def _dss_comparison():
    """Run plot_dss_comparison which fits + compares internally.

    NOTE: Requires DSS with smooth/segmented params — skipped if unavailable.
    """
    from mne_denoise.dss import CombFilterBias

    comb_bias = CombFilterBias(fundamental_freq=LINE_FREQ, sfreq=sfreq)
    return viz.plot_dss_comparison(
        comb_bias,
        raw,
        n_components=10,
        n_select="auto",
        line_freq=LINE_FREQ,
        title="DSS Comparison (sub-01)",
        show=False,
    )


_try("plot_dss_comparison (advanced)", _dss_comparison)


# ══════════════════════════════════════════════════════════════════════
# 6 ── ZapLine module
# ══════════════════════════════════════════════════════════════════════
print("\n── ZapLine ──")

_try(
    "plot_psd_comparison [zapline]",
    viz.zapline.plot_psd_comparison,
    data_before_zap,
    data_after_zap,
    sfreq=sfreq,
    line_freq=LINE_FREQ,
    fmax=100,
    show=False,
    fname=str(ZAP_DIR / "psd_comparison.png"),
)

_try(
    "plot_component_scores [zapline]",
    viz.zapline.plot_component_scores,
    zap,
    show=False,
    fname=str(ZAP_DIR / "component_scores.png"),
)

_try(
    "plot_spatial_patterns [zapline]",
    viz.zapline.plot_spatial_patterns,
    zap,
    show=False,
    fname=str(ZAP_DIR / "spatial_patterns.png"),
)

_try(
    "plot_cleaning_summary [zapline]",
    viz.zapline.plot_cleaning_summary,
    data_before_zap,
    data_after_zap,
    zap,
    sfreq=sfreq,
    line_freq=LINE_FREQ,
    show=False,
)

_try(
    "plot_zapline_summary",
    viz.plot_zapline_summary,
    zap,
    data_before=data_before_zap,
    data_after=data_after_zap,
    sfreq=sfreq,
    channel_names=raw_before_zap.ch_names,
    info=raw_before_zap.info,
    title="ZapLine Summary (sub-01)",
    show=False,
)


# ══════════════════════════════════════════════════════════════════════
# 7 ── Benchmark module
# ══════════════════════════════════════════════════════════════════════
print("\n── Benchmark ──")

# Load pre-computed group metrics from disk
# NOTE: benchmark viz was designed for line-noise metrics (R_f0, etc.)
DERIV = DATA_ROOT / "derivatives" / "mne-denoise"
ln_metrics = DERIV / "group" / "line_noise" / "metrics_all.tsv"
if ln_metrics.exists():
    df_bench = pd.read_csv(ln_metrics, sep="\t")
else:
    # Synthesize minimal test dataframe with line-noise columns
    rows = []
    rng = np.random.default_rng(42)
    for sub in ["sub-01", "sub-02", "sub-03"]:
        for mtag in ["M0", "M1", "M2"]:
            rows.append(
                {
                    "subject": sub,
                    "method": mtag,
                    "R_f0": rng.normal(0.5, 0.2),
                    "peak_attenuation_db": rng.normal(15, 5),
                    "below_noise_pct": rng.normal(0.1, 0.05),
                    "overclean_proportion": rng.normal(0.05, 0.02),
                    "underclean_proportion": rng.normal(0.05, 0.02),
                }
            )
    df_bench = pd.DataFrame(rows)

METHOD_ORDER = sorted(df_bench["method"].unique())
METHOD_COLORS = {
    "M0": "#404040",
    "M1": "#2ca02c",
    "M2": "#d62728",
    "C0": "#404040",
    "C1": "#2ca02c",
    "C2": "#d62728",
}
METHOD_LABELS = {
    "M0": "Baseline",
    "M1": "ZapLine",
    "M2": "DSS-ZapLine",
    "C0": "Baseline",
    "C1": "Paper (ICA)",
    "C2": "DSS (AverageBias)",
}

_try(
    "plot_metric_bars",
    viz.plot_metric_bars,
    df_bench,
    method_order=METHOD_ORDER,
    method_colors=METHOD_COLORS,
    show=False,
    fname=str(BENCH_DIR / "metric_bars.png"),
)

_try(
    "plot_tradeoff_scatter",
    viz.plot_tradeoff_scatter,
    df_bench,
    method_order=METHOD_ORDER,
    method_colors=METHOD_COLORS,
    method_labels=METHOD_LABELS,
    show=False,
    fname=str(BENCH_DIR / "tradeoff_scatter.png"),
)

_try(
    "plot_r_comparison",
    viz.plot_r_comparison,
    df_bench,
    method_order=METHOD_ORDER,
    method_colors=METHOD_COLORS,
    method_labels=METHOD_LABELS,
    show=False,
    fname=str(BENCH_DIR / "r_comparison.png"),
)

_try(
    "plot_paired_metrics",
    viz.plot_paired_metrics,
    df_bench,
    method_order=METHOD_ORDER,
    method_colors=METHOD_COLORS,
    show=False,
    fname=str(BENCH_DIR / "paired_metrics.png"),
)

_try(
    "plot_tradeoff_and_r",
    viz.plot_tradeoff_and_r,
    df_bench,
    method_order=METHOD_ORDER,
    method_colors=METHOD_COLORS,
    method_labels=METHOD_LABELS,
    show=False,
    fname=str(BENCH_DIR / "tradeoff_and_r.png"),
)


# ── PSD-based benchmark plots (need raw freq data) ──
def _make_psd_data():
    """Compute PSD arrays for benchmark gallery/overlay/harmonic funcs."""
    from scipy.signal import welch

    nperseg = min(512, data_before_zap.shape[-1])
    freqs_b, psd_b = welch(data_before_zap, fs=sfreq, nperseg=nperseg, axis=-1)
    gm_before = np.exp(np.mean(np.log(np.maximum(psd_b, 1e-30)), axis=0))

    cleaned_psds = {}
    for tag, d in [("C0", data_before_zap), ("C2", data_after_zap)]:
        freqs_a, psd_a = welch(d, fs=sfreq, nperseg=nperseg, axis=-1)
        gm = np.exp(np.mean(np.log(np.maximum(psd_a, 1e-30)), axis=0))
        cleaned_psds[tag] = (freqs_a, gm)

    return freqs_b, gm_before, cleaned_psds


freqs_b, gm_before, cleaned_psds = _make_psd_data()
nyquist = sfreq / 2.0
harmonics_hz = [LINE_FREQ * (h + 1) for h in range(3) if LINE_FREQ * (h + 1) < nyquist]

_try(
    "plot_qc_psd",
    viz.plot_qc_psd,
    freqs_b,
    gm_before,
    cleaned_psds["C2"][0],
    cleaned_psds["C2"][1],
    method_tag="C2",
    subject=SUB,
    harmonics_hz=harmonics_hz,
    fmax=sfreq / 2,
    show=False,
    fname=str(BENCH_DIR / "qc_psd.png"),
)

_try(
    "plot_psd_gallery",
    viz.plot_psd_gallery,
    freqs_b,
    gm_before,
    cleaned_psds,
    harmonics_hz=harmonics_hz,
    fmax=sfreq / 2,
    subject=SUB,
    method_order=list(cleaned_psds.keys()),
    method_colors=METHOD_COLORS,
    method_labels=METHOD_LABELS,
    show=False,
    fname=str(BENCH_DIR / "psd_gallery.png"),
)

_try(
    "plot_subject_psd_overlay",
    viz.plot_subject_psd_overlay,
    freqs_b,
    gm_before,
    cleaned_psds,
    line_freq=LINE_FREQ,
    fmax=sfreq / 2,
    subject=SUB,
    method_order=list(cleaned_psds.keys()),
    method_colors=METHOD_COLORS,
    method_labels=METHOD_LABELS,
    show=False,
    fname=str(BENCH_DIR / "subject_psd_overlay.png"),
)

_try(
    "plot_harmonic_attenuation",
    viz.plot_harmonic_attenuation,
    freqs_b,
    gm_before,
    cleaned_psds,
    harmonics_hz=harmonics_hz,
    subject=SUB,
    method_order=list(cleaned_psds.keys()),
    method_colors=METHOD_COLORS,
    method_labels=METHOD_LABELS,
    show=False,
    fname=str(BENCH_DIR / "harmonic_attenuation.png"),
)


# ══════════════════════════════════════════════════════════════════════
# 8 ── Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Visual QA Summary")
print("=" * 60)
print(f"  Passed: {len(passed)}/{len(passed) + len(failed)}")
for name in passed:
    print(f"    ✓ {name}")
if failed:
    print(f"\n  Failed: {len(failed)}")
    for name, err in failed:
        print(f"    ✗ {name}: {err}")
print(f"\n  Output: {OUT_DIR}")
print("Done.")
