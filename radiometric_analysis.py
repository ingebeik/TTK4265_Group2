"""
TTK4265 - Radiometric calibration coefficients K (3.1.2) and radiometric response (3.2.2)
========================================================================================
Builds on:
    radiometric_calibration.py   loading, dark subtraction, saturation check
    spectral_calibration.py      pixel -> wavelength (CALIBRATION_POINTS, selected degree)

Run:  python radiometric_analysis.py

Steps:
  1. Per exposure: dark-subtract every frame, convert to counts/s, average    -> S
  2. L(lambda) = L0 * (r/R)^2 * cos(alpha) * rho                               (eq. 1)
  3. K = L / S per pixel; plot K matrix
  4. Uncertainty of K: single frame vs mean / median reference
  5. Response: centre line vs screen, and vs exposure time
"""

import numpy as np
import matplotlib.pyplot as plt

import radiometric_calibration as rc
import spectral_calibration as sc


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = rc.PROJECT_DIR
RESULT_DIR = rc.RESULT_DIR
IRRADIANCE_CSV = PROJECT_DIR / "irradiance_Watts_perm2pernm.csv"

# Eq. (1) geometry
R_M = 1.08        # measured lamp -> screen distance [m]
R_CERT_M = 0.50   # certificate distance r [m]. ASSUMPTION: standard FEL
                  # certificate distance (50 cm). Not in the data files --
                  # confirm against the lamp certificate / with the teacher!
ALPHA_RAD = 0.0   # lamp head-on to the screen centre
RHO = 0.98        # diffuse reflectance of the screen

WL_MIN_NM, WL_MAX_NM = 400.0, 700.0
W_TO_MW = 1000.0  # CSV is in W/(m^2 nm); the assignment writes mW/(m^2 nm)
CENTRE_ROW = 1216 // 2
RESPONSE_WAVELENGTHS_NM = (450, 550, 650)


# ============================================================
# HELPERS
# ============================================================

def wavelength_axis(n_columns):
    """Wavelength of every spectral pixel from the spectral calibration."""
    _, _, models = sc.fit_wavelength_models(sc.CALIBRATION_POINTS)
    model = models[sc.SELECTED_POLYNOMIAL_DEGREE]["model"]
    return model(np.arange(n_columns)), sc.SELECTED_POLYNOMIAL_DEGREE


def screen_radiance(wavelengths_nm):
    """L(lambda) = L0 * (r/R)^2 * cos(alpha) * rho  in mW/(m^2 nm)."""
    certificate = np.loadtxt(IRRADIANCE_CSV, delimiter=",")
    l0 = np.interp(wavelengths_nm, certificate[:, 0], certificate[:, 1]) * W_TO_MW
    return l0 * (R_CERT_M / R_M) ** 2 * np.cos(ALPHA_RAD) * RHO


def load_exposure(exposure_ms, columns):
    """Use radiometric_calibration.process_exposure, keep only what is needed.

    Returns per-frame S (counts/s, saturated pixels = NaN) and the mean
    dark-subtracted counts, both cropped to the spectral window."""
    result = rc.process_exposure(exposure_ms)
    c0, c1 = columns
    seconds = result["exposure_seconds"]

    frames = (result["radiometric_stack"][:, :, c0:c1]
              - result["mean_dark"][None, :, c0:c1]) / seconds
    frames[result["saturation_mask"][:, :, c0:c1]] = np.nan
    frames = frames.astype(np.float32)

    counts = np.nanmean(frames, axis=0) * seconds
    dark_level = float(result["mean_dark"][:, c0:c1].mean())
    return frames, counts, dark_level


# ============================================================
# MAIN
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    n_columns = rc.load_image(rc.IMAGE_DIR / "R200.1.png").shape[1]
    wavelengths, degree = wavelength_axis(n_columns)
    window = np.where((wavelengths >= WL_MIN_NM) & (wavelengths <= WL_MAX_NM))[0]
    c0, c1 = window[0], window[-1] + 1
    wl = wavelengths[c0:c1]
    print(f"\nSpectral model: degree {degree}. "
          f"{WL_MIN_NM:.0f}-{WL_MAX_NM:.0f} nm = columns {c0}..{c1 - 1}")

    L = screen_radiance(wl)
    print(f"L in window: {L.min():.3f} .. {L.max():.3f} mW/(m^2 nm)")

    exposures = rc.EXPOSURE_TIMES_MS
    frames, counts, dark_level = {}, {}, {}
    for e in exposures:
        frames[e], counts[e], dark_level[e] = load_exposure(e, (c0, c1))

    S_mean = {e: np.nanmean(frames[e], axis=0) for e in exposures}   # counts/s

    # Illuminated band from the 200 ms image (no saturation there)
    ref_e = exposures[0]
    row_signal = np.nanmean(S_mean[ref_e], axis=1)
    rows = np.where(row_signal > 0.2 * row_signal.max())[0]
    r0, r1 = rows[0], rows[-1] + 1
    print(f"Illuminated band: rows {r0}..{r1 - 1}, centre row {CENTRE_ROW}")

    # --------------------------------------------------------
    # Raw radiometric calibration frame (mean, dark-corrected, one
    # exposure) -- for visually inspecting vignetting and dust stripes
    # on the slit, cf. the intro text of assignment section 3.2.2.
    # --------------------------------------------------------
    e_show = exposures[1]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    display = np.clip(S_mean[e_show], 0, None)
    image = ax.imshow(display, aspect="auto", cmap="inferno",
                      vmin=0, vmax=np.nanpercentile(display[r0:r1], 99.5),
                      extent=[wl[0], wl[-1], S_mean[e_show].shape[0], 0])
    fig.colorbar(image, label="Dark-corrected signal [counts/s]")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Spatial axis [row]")
    ax.set_title(f"Mean dark-corrected radiometric calibration frame, {e_show} ms")
    fig.savefig(RESULT_DIR / "raw_calibration_frame.png", dpi=120)

    # --------------------------------------------------------
    # Relative radiometric response: each column (wavelength) normalised
    # by its own row-mean. This removes the strong spectral (column-to-
    # column) shape and leaves only the spatial (row-to-row) structure,
    # which is the cleanest way to spot vignetting and dust stripes.
    # --------------------------------------------------------
    column_mean = np.nanmean(S_mean[e_show][r0:r1], axis=0)
    relative_response = S_mean[e_show] / column_mean[None, :]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    image = ax.imshow(relative_response[r0:r1], aspect="auto", cmap="RdBu_r",
                      vmin=0.85, vmax=1.15,
                      extent=[wl[0], wl[-1], r1, r0])
    fig.colorbar(image, label="Signal / column mean")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Spatial axis [row]")
    ax.set_title(f"Relative radiometric response, {e_show} ms "
                 "(reveals vignetting and dust stripes)")
    fig.savefig(RESULT_DIR / "relative_response_map.png", dpi=120)

    # --------------------------------------------------------
    # K = L / S
    # --------------------------------------------------------
    K = {}
    for e in exposures:
        K[e] = L[None, :] / S_mean[e]
        K[e][S_mean[e] <= 0] = np.nan

    K_all = np.nanmean(np.stack([K[e] for e in exposures]), axis=0)
    vmin = np.nanpercentile(K_all[r0:r1], 1)
    vmax = np.nanpercentile(K_all[r0:r1], 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True,
                             constrained_layout=True)
    for ax, e in zip(axes, exposures):
        image = ax.imshow(K[e][r0:r1], aspect="auto", cmap="viridis",
                          vmin=vmin, vmax=vmax,
                          extent=[wl[0], wl[-1], r1, r0])
        ax.set_title(f"K, exposure {e} ms")
        ax.set_xlabel("Wavelength [nm]")
    axes[0].set_ylabel("Spatial axis [row]")
    fig.colorbar(image, ax=axes, label="K [mW m$^{-2}$ nm$^{-1}$ / (counts/s)]")
    fig.savefig(RESULT_DIR / "K_matrix.png", dpi=120)

    centre_medians = [np.nanmedian(K[e][CENTRE_ROW]) for e in exposures]
    print("\nMedian K on the centre row (400-700 nm):")
    for e, m in zip(exposures, centre_medians):
        print(f"  {e} ms: {m:.4e}")
    print(f"Spread between exposures: {max(centre_medians) / min(centre_medians) - 1:.2%}")

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for e in exposures:
        ax.plot(wl, K[e][CENTRE_ROW], label=f"{e} ms")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("K (centre row)")
    ax.set_title("Radiometric coefficient K along the spatial centre line")
    ax.legend()
    fig.savefig(RESULT_DIR / "K_centreline.png", dpi=120)

    # --------------------------------------------------------
    # Uncertainty: single frame vs mean / median reference
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    print("\nUncertainty on the centre row (frame 1 vs reference):")
    for e in exposures:
        K_frames = L[None, :] / frames[e][:, CENTRE_ROW, :]        # (N, cols)
        K_mean = np.nanmean(K_frames, axis=0)
        K_median = np.nanmedian(K_frames, axis=0)
        u_mean = (K_frames[0] - K_mean) / K_mean * 100
        u_median = (K_frames[0] - K_median) / K_median * 100
        spread = np.nanstd(K_frames, axis=0) / K_mean * 100
        axes[0].plot(wl, u_mean, lw=0.7, label=f"{e} ms")
        axes[1].plot(wl, u_median, lw=0.7, label=f"{e} ms")
        print(f"  {e} ms: mean-ref rms {np.sqrt(np.nanmean(u_mean ** 2)):.2f} %, "
              f"median-ref rms {np.sqrt(np.nanmean(u_median ** 2)):.2f} %, "
              f"std over frames {np.nanmean(spread):.2f} %")
    axes[0].set_title("(K$_1$ - K$_{ref}$)/K$_{ref}$, reference = mean of 10 frames "
                      "(centre row)")
    axes[1].set_title("Same, reference = median of 10 frames")
    for ax in axes:
        ax.set_ylabel("Relative deviation [%]")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend()
    axes[1].set_xlabel("Wavelength [nm]")
    fig.savefig(RESULT_DIR / "K_uncertainty.png", dpi=120)

    # Second version: centre line vs a spatially-averaged ("mean of
    # columns") line, since the target is spatially uniform. This is
    # smoother and shows the wavelength-dependent trend more clearly.
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    print("\nUncertainty, centre row vs mean-of-columns (400 ms):")
    e = exposures[-1]
    K_centre = L[None, :] / frames[e][:, CENTRE_ROW, :]
    K_centre_ref = np.nanmean(K_centre, axis=0)
    u_centre = (K_centre[0] - K_centre_ref) / K_centre_ref * 100

    K_band = L[None, :] / np.nanmean(frames[e][:, r0:r1, :], axis=1)   # (N, cols)
    K_band_ref = np.nanmean(K_band, axis=0)
    u_band = (K_band[0] - K_band_ref) / K_band_ref * 100

    ax.plot(wl, u_centre, lw=0.8, label="Centre row")
    ax.plot(wl, u_band, lw=1.4, color="k", label=f"Mean of rows {r0}-{r1 - 1}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Relative deviation [%]")
    ax.set_title(f"K uncertainty at {e} ms: centre row vs spatial mean (reference = mean of 10 frames)")
    ax.legend()
    fig.savefig(RESULT_DIR / "K_uncertainty_centre_vs_band.png", dpi=120)
    print(f"  centre row rms: {np.sqrt(np.nanmean(u_centre ** 2)):.2f} %, "
          f"mean-of-rows rms: {np.sqrt(np.nanmean(u_band ** 2)):.2f} %")

    e = exposures[1]
    K_frames_2d = L[None, None, :] / frames[e]
    K_ref = np.nanmean(K_frames_2d, axis=0)
    u_2d = (K_frames_2d[0] - K_ref) / K_ref * 100
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    image = ax.imshow(u_2d[r0:r1], aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5,
                      extent=[wl[0], wl[-1], r1, r0])
    fig.colorbar(image, label="Deviation [%]")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Spatial axis [row]")
    ax.set_title(f"Single-frame K uncertainty map, {e} ms (reference = mean)")
    fig.savefig(RESULT_DIR / "K_uncertainty_map.png", dpi=120)
    del K_frames_2d

    # --------------------------------------------------------
    # Radiometric response: centre line vs calibration screen
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for e in exposures:
        centre = S_mean[e][CENTRE_ROW]
        ax.plot(wl, centre / np.nanmax(centre), lw=0.9,
                label=f"Sensor, centre line, {e} ms")
    band_mean = np.nanmean(S_mean[ref_e][r0:r1], axis=0)
    ax.plot(wl, band_mean / np.nanmax(band_mean), "k--", lw=0.9,
            label=f"Sensor, mean over rows {r0}-{r1 - 1}, {ref_e} ms")
    ax.plot(wl, L / L.max(), "r", lw=2,
            label="Screen: L$_0$ (r/R)$^2$ $\\rho$ (normalised)")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalised response")
    ax.set_title("Normalised absolute radiometric response vs calibration screen")
    ax.legend(fontsize=8)
    fig.savefig(RESULT_DIR / "response_vs_screen.png", dpi=120)

    # --------------------------------------------------------
    # Response vs exposure time
    # --------------------------------------------------------
    exposure_ms = np.array(exposures, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    print("\nResponse vs exposure (centre row, dark-subtracted counts):")
    for target in RESPONSE_WAVELENGTHS_NM:
        column = int(np.argmin(np.abs(wl - target)))
        y = np.array([counts[e][CENTRE_ROW, column] for e in exposures])
        slope, intercept = np.polyfit(exposure_ms, y, 1)
        fit = slope * exposure_ms + intercept
        r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
        axes[0].plot(exposure_ms, y, "o-", label=f"{wl[column]:.0f} nm")
        print(f"  {wl[column]:.0f} nm: counts = {slope:.2f}*t + {intercept:.1f}, "
              f"R^2 = {r2:.5f}")
    axes[0].set_xlabel("Exposure time [ms]")
    axes[0].set_ylabel("Dark-subtracted counts")
    axes[0].set_title("Sensor response vs exposure time (centre row)")
    axes[0].set_xlim(0, exposure_ms.max() * 1.1)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    axes[1].plot(exposure_ms, [dark_level[e] for e in exposures], "o-", color="k")
    axes[1].set_xlabel("Exposure time [ms]")
    axes[1].set_ylabel("Mean dark level [counts]")
    axes[1].set_title("Dark level vs exposure time (400-700 nm columns)")
    fig.savefig(RESULT_DIR / "response_vs_exposure.png", dpi=120)

    print(f"\nFigures saved in: {RESULT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
