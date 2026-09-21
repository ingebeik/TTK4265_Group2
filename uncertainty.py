"""
TTK4265 Assignment 1, task 3.1.2, second part:
"Plot the uncertainty of the radiometric calibration coefficients, and
comment on the results."

Method from the assignment: radiometrically calibrate a single frame,
subtract it from a reference frame, divide by the reference.

    U_i = (L_ref - L_i) / L_ref,   L_i = K * S_i,   L_ref = K * S_ref

K cancels in that ratio, so U is the frame-to-frame repeatability of the
measurement and does not depend on the absolute scale of K. K is still
applied explicitly, to follow the assignment and to inherit its mask.

Run k_matrix.py first: this reads the K matrix it saves.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
RESULT_DIR = PROJECT_DIR / "results" / "radiometric"

# ------------------------------ USER INPUT ------------------------------
EXPOSURE_MS = 300
NUMBER_OF_IMAGES = 10
SPATIAL_ROW = None    # None -> centre of the rows K was computed on

# Same spectral calibration as k_matrix.py, so the axis matches.
CALIBRATION_POINTS = [
    (267, 404.70), (351, 435.80), (650, 546.10), (737, 578.02),
    (1066, 696.54), (1094, 706.72), (1185, 738.40),
]
POLYNOMIAL_DEGREE = 2
# ------------------------------------------------------------------------


def load_stack(prefix):
    paths = [IMAGE_DIR / f"{prefix}{EXPOSURE_MS}.{i}.png"
             for i in range(1, NUMBER_OF_IMAGES + 1)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing[0]} ...")
    return np.stack([np.asarray(Image.open(p), dtype=np.float64) for p in paths])


def uncertainty(L_single, L_reference):
    """U_i = (L_ref - L_i) / L_ref, one image per frame."""
    return (L_reference[np.newaxis, :, :] - L_single) / L_reference[np.newaxis, :, :]


def main():
    K_path = RESULT_DIR / f"K_{EXPOSURE_MS}ms.npy"
    if not K_path.exists():
        raise FileNotFoundError(f"{K_path} not found. Run k_matrix.py first.")
    K = np.load(K_path)

    screen = load_stack("R")
    dark = load_stack("D")
    exposure_s = EXPOSURE_MS / 1000.0

    # Per-frame count rate, and the two candidate reference frames.
    S_single = (screen - dark.mean(axis=0)[np.newaxis, :, :]) / exposure_s
    S_mean = S_single.mean(axis=0)
    S_median = np.median(S_single, axis=0)

    # Radiometric calibration of each frame and of each reference.
    L_single = K[np.newaxis, :, :] * S_single
    U_mean = uncertainty(L_single, K * S_mean)
    U_median = uncertainty(L_single, K * S_median)

    # Wavelength axis.
    points = np.asarray(CALIBRATION_POINTS, dtype=np.float64)
    model = np.poly1d(np.polyfit(points[:, 0], points[:, 1], POLYNOMIAL_DEGREE))
    wavelengths_nm = model(np.arange(K.shape[1], dtype=np.float64))

    # Spatial line to look at: the centre of the illuminated field, which is
    # the most representative row and is away from the vignetted edges.
    valid_rows = np.where(np.any(np.isfinite(K), axis=1))[0]
    row = SPATIAL_ROW if SPATIAL_ROW is not None else int(np.median(valid_rows))
    columns = np.where(np.isfinite(K[row]))[0]
    lam = wavelengths_nm[columns]

    # RMS over the 10 frames: one uncertainty value per pixel on that row.
    rms_mean = np.sqrt(np.mean(U_mean[:, row, columns] ** 2, axis=0))
    rms_median = np.sqrt(np.mean(U_median[:, row, columns] ** 2, axis=0))

    print(f"\nUncertainty of K, {EXPOSURE_MS} ms, spatial row {row}")
    print(f"Columns used:            {columns[0]}-{columns[-1]} "
          f"({lam[0]:.0f}-{lam[-1]:.0f} nm)")
    print(f"Median |U|, mean ref:    {100 * np.median(rms_mean):.3f} %")
    print(f"Median |U|, median ref:  {100 * np.median(rms_median):.3f} %")
    print(f"Worst |U|, mean ref:     {100 * rms_mean.max():.3f} % "
          f"at {lam[rms_mean.argmax()]:.0f} nm")

    # Summary over every calibrated pixel, not just the one row: the numbers
    # to quote in the report.
    rms_all = np.sqrt(np.mean(U_mean ** 2, axis=0))[np.isfinite(K)]
    print(f"\nOver all {rms_all.size} calibrated pixels (mean reference):")
    print(f"  median:          {100 * np.median(rms_all):.3f} %")
    print(f"  95th percentile: {100 * np.percentile(rms_all, 95):.3f} %")

    # Blue end versus green plateau, to test whether noise is shot-limited.
    blue = (lam >= 400) & (lam <= 430)
    green = (lam >= 520) & (lam <= 600)
    ratio = np.median(rms_mean[blue]) / np.median(rms_mean[green])
    print(f"\n400-430 nm / 520-600 nm uncertainty ratio: {ratio:.2f}")
    print("Correction for frame i being inside the reference: "
          f"divide by sqrt(1 - 1/N) = {np.sqrt(1 - 1 / NUMBER_OF_IMAGES):.3f}")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    for i in range(NUMBER_OF_IMAGES):
        ax.plot(lam, 100 * U_mean[i, row, columns], color="0.8", linewidth=0.5,
                zorder=1, label="Individual frames" if i == 0 else None)

    ax.plot(lam, 100 * rms_mean, color="tab:blue", linewidth=1.6, zorder=3,
            label="RMS over 10 frames, mean reference")
    ax.plot(lam, 100 * rms_median, color="tab:orange", linewidth=1.6, zorder=2,
            label="RMS over 10 frames, median reference")
    ax.axhline(0, color="0.4", linewidth=0.8, zorder=0)

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Relative uncertainty [%]")
    ax.set_title(f"Uncertainty of the radiometric calibration coefficients, "
                 f"spatial row {row}, {EXPOSURE_MS} ms")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(RESULT_DIR / "K_uncertainty.png", dpi=200)
    print(f"\nSaved: {RESULT_DIR / 'K_uncertainty.png'}")
    plt.show()


if __name__ == "__main__":
    main()