"""
TTK4265 Assignment 1, task 3.1.2
"Visualize a matrix of all the K-values. Comment on the results."

Standalone: does its own loading, dark correction and wavelength calibration.
Does not import from the other scripts in the repository.

    K[row, col] = L(lambda(col)) / S[row, col]        [W m^-2 nm^-1 / (counts/s)]

with S the dark-corrected count rate and L the known screen signal, Eq. (1):

    L = L0 * (r/R)^2 * cos(alpha) * rho
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
RESULT_DIR = PROJECT_DIR / "results" / "radiometric"
CERT_FILE = PROJECT_DIR / "data" / "irradiance_Watts_perm2pernm.csv"


# ------------------------------ USER INPUT ------------------------------
EXPOSURE_MS = 300     # exposure time the K matrix is computed from
NUMBER_OF_IMAGES = 10

R_LAB_M = 1.08        # measured lamp-to-screen distance [m]   <-- FILL IN
R_CERT_M = 0.5        # certificate distance [m]
ALPHA_DEG = 0.0       # lamp head-on to the screen, so alpha = 0
RHO = 0.98            # 0.98 per assignment
FIX_560NM = True      # certificate value at 560 nm is ~6 % above its neighbours

# Spectral calibration: identified emission lines (spectral pixel, wavelength).
CALIBRATION_POINTS = [
    (267, 404.70),    # Hg
    (351, 435.80),    # Hg
    (650, 546.10),    # Hg
    (737, 578.02),    # Hg, unresolved doublet
    (1066, 696.54),   # Ar
    (1094, 706.72),   # Ar
    (1185, 738.40),   # Ar
]
POLYNOMIAL_DEGREE = 2
# ------------------------------------------------------------------------

LAMBDA_MIN_NM = 400.0          # usable range of the instrument
LAMBDA_MAX_NM = 700.0
SATURATION_VALUE = 65535       # 16-bit full well
ILLUMINATION_FRACTION = 0.25   # finds the rows the screen lit
ROW_MARGIN = 15                # rows dropped at each edge of that field,
                               # where the screen only partly lit the slit


def load_stack(prefix):
    """Load the 10 frames of one series as a 3D array, checking them."""
    paths = [IMAGE_DIR / f"{prefix}{EXPOSURE_MS}.{i}.png"
             for i in range(1, NUMBER_OF_IMAGES + 1)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing[0]} ...")

    images = []
    for path in paths:
        image = np.asarray(Image.open(path), dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(
                f"{path.name} has shape {image.shape}; expected a "
                f"two-dimensional grayscale image"
            )
        if images and image.shape != images[0].shape:
            raise ValueError(
                f"{path.name} has shape {image.shape}, "
                f"expected {images[0].shape}"
            )
        images.append(image)

    stack = np.stack(images)
    print(f"{prefix}{EXPOSURE_MS}: {len(images)} frames, shape {stack.shape[1:]}, "
          f"range {stack.min():.0f}-{stack.max():.0f}")
    return stack


def main():
    if R_LAB_M is None:
        raise ValueError("Set R_LAB_M (measured lamp-screen distance) first.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # --- S: dark-corrected count rate [counts/s] ---
    screen = load_stack("R")
    dark = load_stack("D")
    exposure_s = EXPOSURE_MS / 1000.0
    S = (screen.mean(axis=0) - dark.mean(axis=0)) / exposure_s

    # --- wavelength of each column (smile ignored) ---
    points = np.asarray(CALIBRATION_POINTS, dtype=np.float64)
    wavelength_model = np.poly1d(
        np.polyfit(points[:, 0], points[:, 1], POLYNOMIAL_DEGREE)
    )
    wavelengths_nm = wavelength_model(np.arange(S.shape[1], dtype=np.float64))

    # --- L: known screen signal, Eq. (1) ---
    # Identical for every row: the screen is uniform along the slit, so all
    # row-to-row structure in K originates in the instrument.
    cert = np.loadtxt(CERT_FILE, delimiter=",")
    if FIX_560NM:
        i = np.where(np.isclose(cert[:, 0], 560.0))[0]
        if i.size:
            cert[i[0], 1] = 0.5 * (cert[i[0] - 1, 1] + cert[i[0] + 1, 1])
    L0 = np.interp(wavelengths_nm, cert[:, 0], cert[:, 1])

    if RHO is None:
        refl = np.loadtxt(REFL_FILE)
        rho = np.interp(wavelengths_nm, refl[:, 0], refl[:, 1])
    else:
        rho = RHO

    L = L0 * (R_CERT_M / R_LAB_M) ** 2 * np.cos(np.deg2rad(ALPHA_DEG)) * rho

    # --- keep only pixels where K is meaningful ---
    in_range = (wavelengths_nm >= LAMBDA_MIN_NM) & (wavelengths_nm <= LAMBDA_MAX_NM)

    row_signal = np.median(S[:, in_range], axis=1)
    lit = np.where(row_signal > ILLUMINATION_FRACTION * row_signal.max())[0]
    first_row, last_row = lit[0] + ROW_MARGIN, lit[-1] - ROW_MARGIN
    rows = np.zeros(S.shape[0], dtype=bool)
    rows[first_row:last_row + 1] = True

    saturated = (screen >= SATURATION_VALUE).any(axis=0)   # clipping inflates K
    valid = in_range[np.newaxis, :] & rows[:, np.newaxis] & ~saturated

    # --- K ---
    K = np.full(S.shape, np.nan)
    K[valid] = np.broadcast_to(L, S.shape)[valid] / S[valid]
    np.save(RESULT_DIR / f"K_{EXPOSURE_MS}ms.npy", K)

    print(f"\nK matrix, {EXPOSURE_MS} ms")
    print(f"Rows used:                {first_row}-{last_row}")
    print(f"Saturated pixels removed: {saturated.sum()}")
    print(f"Valid pixels:             {np.isfinite(K).sum()} of {K.size}")
    print(f"K min:                    {np.nanmin(K):.3e}")
    print(f"K max:                    {np.nanmax(K):.3e}")
    print(f"K median:                 {np.nanmedian(K):.3e}"
          "  [W m^-2 nm^-1 / (counts/s)]")

    # --- the figure the task asks for ---
    columns = np.where(in_range)[0]
    K_crop = np.ma.masked_invalid(
        K[first_row:last_row + 1, columns[0]:columns[-1] + 1]
    )

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    image = ax.imshow(
        K_crop, aspect="auto", cmap="viridis",
        vmin=np.percentile(K_crop.compressed(), 1),
        vmax=np.percentile(K_crop.compressed(), 99),
        extent=[wavelengths_nm[columns[0]], wavelengths_nm[columns[-1]],
                last_row, first_row],
    )
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Spatial pixel")
    ax.set_title(f"Radiometric calibration coefficients K, {EXPOSURE_MS} ms")
    fig.colorbar(image, ax=ax,
                 label=r"K [W m$^{-2}$ nm$^{-1}$ / (counts s$^{-1}$)]")

    fig.savefig(RESULT_DIR / "K_matrix.png", dpi=200)
    print(f"\nSaved: {RESULT_DIR / 'K_matrix.png'}")
    plt.show()


if __name__ == "__main__":
    main()