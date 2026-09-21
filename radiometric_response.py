"""
TTK4265 Assignment 1, task 3.2.2:
"Plot a normalized absolute radiometric response of the spatial center line
of the image alongside the radiometric response of the calibration screen
and comment on the results."

Follows Henriksen et al. 2022 (HYPSO-1 pre-launch calibration), Fig. 7:
the absolute radiometric response is the average pixel response per
wavelength, plotted against the reference signal from the halogen lamp.
Smile is ignored, as the assignment allows.

Standalone: needs only the lab images and the certificate files.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
RESULT_DIR = PROJECT_DIR / "results" / "radiometric"
CERT_FILE = PROJECT_DIR / "data" / "irradiance_Watts_perm2pernm.csv"
REFL_FILE = PROJECT_DIR / "data" / "reflectances.txt"

# ------------------------------ USER INPUT ------------------------------
EXPOSURE_MS = 300
NUMBER_OF_IMAGES = 10

CENTRE_ROW = 1216 // 2   # "spatial center line of the image"
PROFILE_HALF_WIDTH = 5   # median of 11 rows, to suppress pixel noise

RHO = 0.98               # None -> reflectances.txt
FIX_560NM = True         # certificate value at 560 nm is ~6 % too high
SHOW_RELATIVE = False    # response / screen, the instrument's own spectral
                         # efficiency. Not required by the task.
SHOW_COLUMN_MEAN = False # the same response as the mean of the columns.
                         # Not required: the task asks for the centre line.

CALIBRATION_POINTS = [
    (267, 404.70), (351, 435.80), (650, 546.10), (737, 578.02),
    (1066, 696.54), (1094, 706.72), (1185, 738.40),
]
POLYNOMIAL_DEGREE = 2
# ------------------------------------------------------------------------

LAMBDA_MIN_NM = 400.0
LAMBDA_MAX_NM = 700.0
ILLUMINATION_FRACTION = 0.25
ROW_MARGIN = 15


def load_stack(prefix):
    paths = [IMAGE_DIR / f"{prefix}{EXPOSURE_MS}.{i}.png"
             for i in range(1, NUMBER_OF_IMAGES + 1)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing[0]} ...")
    return np.stack([np.asarray(Image.open(p), dtype=np.float64) for p in paths])


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    screen = load_stack("R")
    dark = load_stack("D")
    S = (screen.mean(axis=0) - dark.mean(axis=0)) / (EXPOSURE_MS / 1000.0)

    # Wavelength per column.
    points = np.asarray(CALIBRATION_POINTS, dtype=np.float64)
    model = np.poly1d(np.polyfit(points[:, 0], points[:, 1], POLYNOMIAL_DEGREE))
    wavelengths_nm = model(np.arange(S.shape[1], dtype=np.float64))
    in_range = (wavelengths_nm >= LAMBDA_MIN_NM) & (wavelengths_nm <= LAMBDA_MAX_NM)
    columns = np.where(in_range)[0]
    lam = wavelengths_nm[columns]

    # (a) Absolute radiometric response of the spatial centre line.
    first = CENTRE_ROW - PROFILE_HALF_WIDTH
    last = CENTRE_ROW + PROFILE_HALF_WIDTH + 1
    centre_line = np.median(S[first:last, :], axis=0)[columns]

    # (b) Same quantity as the mean of the columns, over the illuminated rows
    #     only. This is the definition used in the assignment text and in
    #     HYPSO Fig. 7; it differs from (a) only through vignetting.
    row_signal = np.median(S[:, in_range], axis=1)
    lit = np.where(row_signal > ILLUMINATION_FRACTION * row_signal.max())[0]
    lit_rows = slice(lit[0] + ROW_MARGIN, lit[-1] - ROW_MARGIN + 1)
    column_mean = S[lit_rows, :].mean(axis=0)[columns]

    # (c) Radiometric response of the calibration screen: the known signal.
    cert = np.loadtxt(CERT_FILE, delimiter=",")
    if FIX_560NM:
        i = np.where(np.isclose(cert[:, 0], 560.0))[0]
        if i.size:
            cert[i[0], 1] = 0.5 * (cert[i[0] - 1, 1] + cert[i[0] + 1, 1])
    L0 = np.interp(lam, cert[:, 0], cert[:, 1])
    if RHO is None:
        refl = np.loadtxt(REFL_FILE)
        rho = np.interp(lam, refl[:, 0], refl[:, 1])
    else:
        rho = RHO
    screen_response = L0 * rho     # (r/R)^2 cos(alpha) is a constant scale
                                   # factor and drops out on normalisation

    def normalise(x):
        return x / np.nanmax(x)

    centre_n = normalise(centre_line)
    column_n = normalise(column_mean)
    screen_n = normalise(screen_response)

    print(f"\nAbsolute radiometric response, {EXPOSURE_MS} ms")
    print(f"Centre line: rows {first}-{last - 1}")
    print(f"Column mean: rows {lit_rows.start}-{lit_rows.stop - 1}")
    print(f"Response peaks at {lam[np.argmax(centre_n)]:.0f} nm; "
          f"screen peaks at {lam[np.argmax(screen_n)]:.0f} nm")
    print("\n  lam   response   screen   response/screen")
    relative = normalise(centre_line / screen_response)
    for target in (400, 450, 500, 550, 600, 650, 700):
        if lam[0] <= target <= lam[-1]:
            j = int(np.argmin(np.abs(lam - target)))
            print(f"  {target}    {centre_n[j]:.3f}     {screen_n[j]:.3f}"
                  f"      {relative[j]:.3f}")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(lam, centre_n, color="tab:blue", linewidth=1.6,
            label="Absolute radiometric response, spatial centre line")
    if SHOW_COLUMN_MEAN:
        ax.plot(lam, column_n, color="tab:blue", linewidth=1.0, linestyle="--",
                label="Absolute radiometric response, mean of columns")
    ax.plot(lam, screen_n, color="tab:orange", linewidth=1.6,
            label="Radiometric response of the calibration screen")
    if SHOW_RELATIVE:
        ax.plot(lam, relative, color="tab:green", linewidth=1.2, linestyle=":",
                label="Response / screen (relative spectral efficiency)")

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized response [-]")
    ax.set_title(f"Normalized absolute radiometric response, {EXPOSURE_MS} ms")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(RESULT_DIR / "radiometric_response.png", dpi=200)
    print(f"\nSaved: {RESULT_DIR / 'radiometric_response.png'}")
    plt.show()


if __name__ == "__main__":
    main()