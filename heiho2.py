"""
TTK4265 - Spectral Calibration (Section 3.1.1)
================================================
Assumes this script sits INSIDE the same folder as your PNG images, e.g.:
    your_project/
        Images/
            spectral_calibration.py   <- this file
            AR-1.300.png   (argon lamp, exposure "300")
            HG-1.300.png   (mercury lamp, exposure "300")
            ...

Note on lamp naming: files are LAMP-<id>.<exposure>.png. AR = argon tube,
HG = mercury tube. The mercury tube's spectrogram also shows the full
argon-like peak cluster (pixels ~1065-1702) on top of its own 3 isolated
lines -- this is expected, since many "mercury" calibration lamps use
argon as a buffer/fill gas that also emits its own spectrum (this is the
same reason the reference papers cite a combined "mercury-argon" lamp
model). Only ONE frame per lamp was captured (confirmed OK by course
staff), so no frame-averaging step is included here.

Steps:
  1. Load Ar and Hg spectrograms, extract center-row profiles
  2. Create a combined image (sum) containing lines from both lamps
  3. Detect peaks, pair with known reference wavelengths
  4. Fit polynomials (orders 1-4), evaluate with leave-one-out RMSE
  5. Report the best-fitting model
"""

import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

IMAGES_DIR = 'Images'   # script sits in labbi/, images are in labbi/Images/
OUTPUT_DIR = '.'

# Best (highest-SNR, non-saturated) exposure of each lamp from your dataset
AR_IMAGE = f'{IMAGES_DIR}/AR-1.300.png'   # argon tube, exposure 300
HG_IMAGE = f'{IMAGES_DIR}/HG-1.300.png'   # mercury tube, exposure 300

# --- Argon (Ar I) reference wavelengths, nm ---
# From Henriksen, Sigernes & Johansen (2022), "A Closer Look At A
# Spectrographic Wavelength Calibration", Table 2 (same Newport 6030 lamp).
# Doublets too close for this instrument's FWHM to resolve are averaged.
AR_REFERENCE_NM = [
    696.54, 706.72, 727.29, 738.40, 750.93, 763.51, 772.395,
    794.82, 801.05, 810.95, 826.45, 841.635, 852.14, 912.30,
]
# Pixel positions of the 14 Ar peaks IN ORDER, from AR-1.300.png
# (confirmed in earlier analysis: found via find_peaks, height>1500,
#  prominence>800 -- verify against your own plot before trusting this!)
AR_PIXELS = [1065, 1094, 1152, 1184, 1220, 1257, 1282, 1348,
             1367, 1396, 1441, 1488, 1519, 1702]

# --- Mercury (Hg I) reference wavelengths, nm ---
# The 3 isolated strong peaks present in HG-1/HG-2 but NOT in AR-1.
# 577.0/579.1 doublet is unresolved at this instrument's FWHM -> averaged.
HG_REFERENCE_NM = [435.8, 546.1, 578.05]
HG_PIXELS = [351, 650, 737]   # verify against your own plot!

POLY_ORDERS = [1, 2, 3, 4]


# ============================================================
# STEP 1 -- load + combine
# ============================================================

def load_image(path):
    return np.array(Image.open(path)).astype(float)


def center_row(img):
    return img[img.shape[0] // 2, :]


def make_combined_image(img_a, img_b):
    """Sum two spectrograms pixel-wise (per assignment: 'Create a combined
    image containing emission lines from all, or at least two lamp
    sources, e.g. by addition')."""
    return img_a + img_b


# ============================================================
# STEP 2 -- peak detection (for verification / visualization)
# ============================================================

def detect_peaks(profile, height, distance, prominence):
    smoothed = uniform_filter1d(profile, size=3)
    peaks, _ = find_peaks(smoothed, height=height, distance=distance,
                           prominence=prominence)
    return np.sort(peaks), smoothed


def plot_profile(smoothed, peaks, title, out_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(smoothed, lw=0.8, color='steelblue')
    ax.plot(peaks, smoothed[peaks], 'rx', markersize=7)
    for p in peaks:
        ax.annotate(str(p), (p, smoothed[p]), textcoords="offset points",
                    xytext=(0, 6), fontsize=6, rotation=90)
    ax.set_xlabel('Pixel index (spectral axis)')
    ax.set_ylabel('Intensity [counts]')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================
# STEP 3 -- leave-one-out RMSE across polynomial orders
# ============================================================

def loo_rmse(pixels, wavelengths, order):
    n = len(pixels)
    errors = []
    for i in range(n):
        train_p = np.delete(pixels, i)
        train_w = np.delete(wavelengths, i)
        coeffs = np.polyfit(train_p, train_w, order)
        pred = np.polyval(coeffs, pixels[i])
        errors.append(wavelengths[i] - pred)
    errors = np.array(errors)
    return np.sqrt(np.mean(errors**2)), errors


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    ar_img = load_image(AR_IMAGE)
    hg_img = load_image(HG_IMAGE)

    ar_profile = center_row(ar_img)
    hg_profile = center_row(hg_img)

    # --- Combined image + profile (assignment's explicit "combined image" task) ---
    combined_img = make_combined_image(ar_img, hg_img)
    combined_profile = center_row(combined_img)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.clip(combined_img, 0, np.percentile(combined_img, 99.5)),
               aspect='auto', cmap='inferno')
    plt.colorbar(label='Counts')
    plt.xlabel('Spectral axis [pixel]')
    plt.ylabel('Spatial axis [pixel]')
    plt.title('Combined Ar + Hg-Ar spectrogram (sum)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/combined_spectrogram.png', dpi=120)
    plt.close()

    combined_peaks, combined_smoothed = detect_peaks(
        combined_profile, height=1000, distance=5, prominence=500)
    plot_profile(combined_smoothed, combined_peaks,
                 'Combined Ar+Hg profile -- verify peaks before trusting the fit!',
                 f'{OUTPUT_DIR}/combined_profile_peaks.png')
    print(f"Combined profile: {len(combined_peaks)} peaks detected "
          f"(sanity check against AR_PIXELS + HG_PIXELS below)")

    # --- Build the full (pixel, wavelength) calibration set ---
    pixels = np.array(HG_PIXELS + AR_PIXELS, dtype=float)
    wavelengths = np.array(HG_REFERENCE_NM + AR_REFERENCE_NM)

    order_idx = np.argsort(pixels)
    pixels = pixels[order_idx]
    wavelengths = wavelengths[order_idx]

    print(f"\nTotal calibration points: {len(pixels)} "
          f"({len(HG_PIXELS)} Hg + {len(AR_PIXELS)} Ar)")
    print("\npixel   wavelength (nm)")
    for p, w in zip(pixels, wavelengths):
        print(f"{p:6.0f}  {w:8.3f}")

    # --- RMSE table ---
    print("\n=== RMSE (leave-one-out cross-validation) ===")
    print(f"{'Order':<8}{'RMSE (nm)':<12}")
    results = {}
    for order in POLY_ORDERS:
        rmse, _ = loo_rmse(pixels, wavelengths, order)
        results[order] = rmse
        print(f"{order:<8}{rmse:<12.4f}")

    best_order = min(results, key=results.get)
    print(f"\nLowest RMSE: order {best_order} ({results[best_order]:.4f} nm)")

    final_coeffs = np.polyfit(pixels, wavelengths, best_order)
    print(f"\nFinal calibration coefficients (order {best_order}, "
          f"highest power first):\n{final_coeffs}")

    # ========================================================
    # Find the pixel window corresponding to 400-700 nm
    # ========================================================
    # Evaluate the fitted polynomial densely across the full pixel range,
    # then find which pixel gives the wavelength closest to 400 and 700 nm.
    pixel_grid = np.arange(0, 1936, 1)
    wavelength_grid = np.polyval(final_coeffs, pixel_grid)

    pixel_at_400 = pixel_grid[np.argmin(np.abs(wavelength_grid - 400))]
    pixel_at_700 = pixel_grid[np.argmin(np.abs(wavelength_grid - 700))]
    print(f"\n=== Pixel window for 400-700 nm (using order {best_order} fit) ===")
    print(f"400 nm -> pixel {pixel_at_400}")
    print(f"700 nm -> pixel {pixel_at_700}")
    print(f"(This is the pixel range to restrict later analysis "
          f"-- radiometric response, FWHM, etc. -- to.)")

    # ========================================================
    # Per-point leave-one-out residuals, for every order
    # -- lets you see WHICH points are driving each order's RMSE,
    #    rather than just the aggregate number.
    # ========================================================
    fig, axes = plt.subplots(len(POLY_ORDERS), 1, figsize=(10, 3*len(POLY_ORDERS)),
                              sharex=True)
    all_errors = {}
    for ax, order in zip(axes, POLY_ORDERS):
        rmse, errors = loo_rmse(pixels, wavelengths, order)
        all_errors[order] = errors
        colors = ['red' if w in (852.14, 912.30) else 'steelblue'
                  for w in wavelengths]
        ax.bar(range(len(pixels)), errors, color=colors)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel(f'Order {order}\nresidual (nm)')
        ax.set_title(f'Order {order}: RMSE={rmse:.3f} nm '
                     f'(red = 852.14 & 912.30 nm points)')
    axes[-1].set_xticks(range(len(pixels)))
    axes[-1].set_xticklabels([f'{w:.1f}' for w in wavelengths], rotation=90, fontsize=7)
    axes[-1].set_xlabel('Reference wavelength (nm)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/loo_residuals_per_point.png', dpi=120)
    plt.close(fig)

    print("\n=== Per-point LOO residuals (nm) by order ===")
    header = "wavelength(nm) " + "  ".join(f"order{o}" for o in POLY_ORDERS)
    print(header)
    for i, w in enumerate(wavelengths):
        row = f"{w:14.3f} " + "  ".join(f"{all_errors[o][i]:+7.3f}" for o in POLY_ORDERS)
        flag = "  <-- far-red point" if w in (852.14, 912.30) else ""
        print(row + flag)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(results.keys()), list(results.values()), 'o-')
    ax.set_xlabel('Polynomial order')
    ax.set_ylabel('RMSE (nm), leave-one-out')
    ax.set_title('Combined Hg+Ar calibration: RMSE vs polynomial order')
    ax.set_xticks(POLY_ORDERS)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rmse_vs_order_combined.png', dpi=120)
    plt.close(fig)

    print(f"\nSaved: combined_spectrogram.png, combined_profile_peaks.png, "
          f"rmse_vs_order_combined.png")