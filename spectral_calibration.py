from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR/"Images"
RESULT_DIR = PROJECT_DIR/"results"/"spectral"

EXPOSURE_MS = 300

LAMP_FILES = {
    "Argon": IMAGE_DIR / "Argon" / "AR-1.300.png",
    "Mercury": IMAGE_DIR / "Mercury" / "HG-1.300.png",
    "CO2": IMAGE_DIR / "CO2" / "CO2-1.300.png",
}

#Images have 1216 spatial rows (vertical) and 1936 spectral columns (horiz)
CENTRE_ROW = 1216//2
#608 +- 5. use 11 rows instead of only 1, to account for random pixel noise
PROFILE_HALF_WIDTH = 5

#max saturation value in unsigned 16-bit image
SATURATION_VALUE = 65535

PEAK_PROMINENCE = {
    "Argon": 300,
    "Mercury": 300,
    "CO2": 150,
}

CALIBRATION_POINTS = [
    ("Mercury", 267, 404.7),
    ("Mercury", 351, 435.8),
    ("Mercury", 650, 546.1),
    ("Mercury", 737, 578.02),  #Unresolved Hg doublet!!
    ("Argon", 1066, 696.54),
    ("Argon", 1094, 706.72),
    ("Argon", 1185, 738.40)
]

POLYNOMIAL_DEGREES = [1, 2, 3]


SLIT_WIDTH_M = 25e-6
COLLIMATOR_FOCAL_LENGTH_M = 30e-3
GRATING_GROOVE_DENSITY_PER_M = 600e3
GRATING_INCIDENCE_ANGLE_DEG = 0

FWHM_LINES = [
    {
        "lamp": "Mercury",
        "approximate_pixel": 351,
        "wavelength_nm": 435.8,
    },
    {
        "lamp": "Mercury",
        "approximate_pixel": 650,
        "wavelength_nm": 546.1,
    },
    {
        "lamp": "Argon",
        "approximate_pixel": 1066,
        "wavelength_nm": 696.54,
    },
]

#Eval every fifth spatial row in the illuminated region
FWHM_SPATIAL_ROWS = np.arange(400, 851, 5)

#Search for each peak within ±18 pixels of its expected position
FWHM_SEARCH_HALF_WIDTH = 18
SELECTED_POLYNOMIAL_DEGREE = 2 #bruker andreordens

def load_image(path):
    if not path.exists():
        raise FileNotFoundError(f"could not find image: {path}")
    image = np.asarray(Image.open(path), dtype = np.float64)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional grayscale image, "
            f"but {path.name} has shape {image.shape}"
        )
    return image
def load_mean_dark_image(exposure_ms):
    """ Load and average all dark images with requested exposure rate"""

    dark_paths = sorted(IMAGE_DIR.glob(f"D{exposure_ms}.*.png"))
    if len(dark_paths) == 0:
        raise FileNotFoundError(f"No D{exposure_ms} dark images were found in {IMAGE_DIR}")
    dark_images = [load_image(path) for path in dark_paths]
    expected_shape = dark_images[0].shape
    for path, image in zip(dark_paths, dark_images):
        if image.shape != expected_shape:
            raise ValueError(f"{path.name} has shape {image.shape}, expected {expected_shape})")
    dark_stack = np.stack(dark_images, axis=0)
    mean_dark = np.mean(dark_stack, axis=0)

    return mean_dark, dark_paths

def extract_centre_profile(image, centre_row = CENTRE_ROW, half_width = PROFILE_HALF_WIDTH):
    """ extract 1D spectrum near the spatial centre, median of 11 neighbouring rows is used ro reduce effects of random pixel noise"""
    first_row= centre_row -half_width
    last_row = centre_row + half_width +1
    if first_row <0 or last_row > image.shape[0]:
        raise ValueError("The selected row interval is outside the image")
    centre_region = image[first_row:last_row,:]
    #collapse spatial dir, leaving only the spectral axis
    profile = np.median(centre_region, axis=0)
    return profile
def print_img_info(name, raw_image, corrected_image):
    saturated_pixels = np.count_nonzero(raw_image >= SATURATION_VALUE)
    saturated_percentage = 100*saturated_pixels/raw_image.size

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Image shape: {raw_image.shape}")
    print(f"Raw minimum: {raw_image.min():.1f}")
    print(f"Raw maximum: {raw_image.max():.1f}")
    print(f"Raw median: {np.median(raw_image):.1f}")
    print(f"Corrected minimum: {corrected_image.min():.1f}")
    print(f"Corrected maximum: {corrected_image.max():.1f}")
    print(f"Saturated pixels: {saturated_pixels}")
    print(f"Saturated percentage: {saturated_percentage:.6f} %")

def plot_corrected_images(corrected_images):
    fig,axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize =(14,12),
        constrained_layout=True)
    for ax, (name, image) in zip(axes, corrected_images.items()):
        display_image = np.maximum(image, 0)
        display_maximum = np.percentile(display_image, 99.9) #prevents a few very bright pixels from making the rest of the pic appear completely black
        shown_image = ax.imshow(
            display_image,
            cmap="gray",
            aspect="auto",
            vmin=0,
            vmax= display_maximum
        )
        ax.set_title(f"{name}, dark-corrected, {EXPOSURE_MS} ms")
        ax.set_xlabel("Spectral pixel")
        ax.set_ylabel("Spatial pixel")
        ax.legend(loc="upper right")

        fig.colorbar(
            shown_image,
            ax=ax,
            label = "Dark-corrected counts"
        )
    output_path = RESULT_DIR/"corrected_lamp_images.png"
    fig.savefig(output_path, dpi=200)
    return fig
def plot_centre_profiles(profiles):
    fig,axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(15,19),
        sharex=True,
        constrained_layout=True
    )
    for ax, (name,profile) in zip(axes, profiles.items()):
        spectral_pixels = np.arange(profile.size)
        ax.plot(
            spectral_pixels,
            profile,
            linewidth = 0.8
        )
        ax.set_title(f"{name}. centre spectral profile")
        ax.set_ylabel("dark-corrected counts")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Spectral pixel")
    output_path = RESULT_DIR/"centre_spectral_profiles.png"
    fig.savefig(output_path, dpi=200)
    return fig

def normalize_for_combination(image):
    """
    Normalize one dark-corrected image for the combined display.

    Each lamp is normalized independently because CO2 is much weaker
    than Hg and Ar. The resulting image shows spectral-line positions,
    but not the true relative intensity between the lamps.
    """
    positive_image = np.maximum(image, 0)

    scale = np.percentile(
        positive_image,
        99.9
    )

    if scale <= 0:
        raise ValueError(
            "Cannot normalize an image without positive signal"
        )

    return positive_image / scale


def plot_combined_image(corrected_images):
    """Create combined image containing Ar, Hg and CO2 features """
    normalized_images = [
        normalize_for_combination(image)
        for image in corrected_images.values()
    ]

    combined_image = np.sum(
        normalized_images,
        axis=0
    )

    fig, ax = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True
    )

    shown_image = ax.imshow(
        combined_image,
        cmap="inferno",
        aspect="auto",
        vmin=0,
        vmax=np.percentile(combined_image, 99.9)
    )

    ax.axhline(
        CENTRE_ROW,
        color="cyan",
        linestyle="--",
        linewidth=1,
        label=f"Centre row: {CENTRE_ROW}"
    )

    ax.set_title(
        "Combined normalized Ar, Hg and CO2 spectra"
    )
    ax.set_xlabel("Spectral pixel")
    ax.set_ylabel("Spatial pixel")
    ax.legend(loc="upper right")

    fig.colorbar(
        shown_image,
        ax=ax,
        label="Combined normalized intensity"
    )

    output_path = RESULT_DIR / "combined_lamp_image.png"
    fig.savefig(output_path, dpi=200)

    return fig

def detect_peaks_in_profiles(profiles):
    """ Detect candidate peaks and show their spectral pixels """
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(16, 12),
        sharex=True,
        constrained_layout=True
    )

    detected_peaks = {}

    for ax, (name, profile) in zip(
        axes,
        profiles.items()
    ):
        peaks, properties = find_peaks(
            profile,
            prominence=PEAK_PROMINENCE[name],
            distance=8,
            width=2
        )

        detected_peaks[name] = peaks
        spectral_pixels = np.arange(profile.size)

        ax.plot(
            spectral_pixels,
            profile,
            linewidth=0.8,
            label="Measured profile"
        )

        ax.scatter(
            peaks,
            profile[peaks],
            color="red",
            marker="x",
            s=45,
            label="Detected candidate"
        )

        for peak in peaks:
            ax.annotate(
                str(peak),
                xy=(peak, profile[peak]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=90
            )

        ax.set_title(f"{name}: detected candidate peaks")
        ax.set_ylabel("Dark-corrected counts")
        ax.grid(alpha=0.25)
        ax.legend()

        print(f"\n{name}")
        print(f"Detected peaks: {peaks.tolist()}")

    axes[-1].set_xlabel("Spectral pixel")

    output_path = RESULT_DIR / "detected_spectral_peaks.png"
    fig.savefig(output_path, dpi=200)

    return fig, detected_peaks

def fit_wavelength_models(calibration_points):
    """ Fit polynomial models mapping spectral pixel to wavelength  """
    pixels = np.array(
        [point[1] for point in calibration_points],
        dtype=np.float64
    )

    wavelengths = np.array(
        [point[2] for point in calibration_points],
        dtype=np.float64
    )

    models = {}

    print("\nWavelength calibration")
    print("======================")
    print("\nCalibration points:")

    for lamp, pixel, wavelength in calibration_points:
        print(
            f"{lamp:10s}: pixel {pixel:4d} "
            f"-> {wavelength:7.2f} nm"
        )

    print("\nPolynomial fitting results:")

    for degree in POLYNOMIAL_DEGREES:
        coefficients = np.polyfit(
            pixels,
            wavelengths,
            degree
        )

        model = np.poly1d(coefficients)

        predicted_wavelengths = model(pixels)

        residuals = (
            wavelengths
            - predicted_wavelengths
        )

        rmse = np.sqrt(
            np.mean(residuals**2)
        )

        models[degree] = {
            "model": model,
            "coefficients": coefficients,
            "predicted_wavelengths": predicted_wavelengths,
            "residuals": residuals,
            "rmse": rmse,
        }

        print(
            f"Degree {degree}: RMSE = {rmse:.4f} nm"
        )

    return pixels, wavelengths, models

def plot_wavelength_models(
    calibration_points,
    pixels,
    wavelengths,
    models
):
    """ Plot the calibration points and fitted wavelength models"""
    fig, ax = plt.subplots(
        figsize=(10, 7),
        constrained_layout=True
    )

    lamp_colours = {
        "Mercury": "blue",
        "Argon": "red",
        "CO2": "green",
    }

    #Plot each calibration point
    for lamp, pixel, wavelength in calibration_points:
        ax.scatter(
            pixel,
            wavelength,
            color=lamp_colours[lamp],
            s=60,
            zorder=3
        )

        ax.annotate(
            f"{lamp}, {wavelength:.1f} nm",
            xy=(pixel, wavelength),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    pixel_range = np.linspace(
        pixels.min(),
        pixels.max(),
        500
    )

    for degree, result in models.items():
        model = result["model"]
        rmse = result["rmse"]

        ax.plot(
            pixel_range,
            model(pixel_range),
            linewidth=1.5,
            label=(
                f"Degree {degree}, "
                f"RMSE = {rmse:.3f} nm"
            )
        )

    ax.set_xlabel("Spectral pixel")
    ax.set_ylabel("Wavelength [nm]")
    ax.set_title("Pixel-to-wavelength calibration")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path = RESULT_DIR / "wavelength_calibration_models.png"
    fig.savefig(output_path, dpi=200)

    return fig
def plot_calibration_rmse(models):
    """Plot calibration RMSE against polynomial degree"""
    degrees = list(models.keys())

    rmse_values = [
        models[degree]["rmse"]
        for degree in degrees
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5),
        constrained_layout=True
    )

    ax.plot(
        degrees,
        rmse_values,
        marker="o"
    )

    ax.set_xticks(degrees)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE [nm]")
    ax.set_title("Wavelength-calibration fitting error")
    ax.grid(alpha=0.25)

    output_path = RESULT_DIR / "wavelength_calibration_rmse.png"
    fig.savefig(output_path, dpi=200)

    return fig
def calculate_theoretical_fwhm(order):
    """Calculate theoretical slit-limited spectral FWHM.
    FWHM = groove_spacing* cos(alpha)*slit_width/(order*collimator_focal_length)"""

    groove_spacing_m = (
        1.0 / GRATING_GROOVE_DENSITY_PER_M
    )

    incidence_angle_rad = np.deg2rad(
        GRATING_INCIDENCE_ANGLE_DEG
    )

    fwhm_m = (
        groove_spacing_m*np.cos(incidence_angle_rad)*SLIT_WIDTH_M/(order*COLLIMATOR_FOCAL_LENGTH_M)
    )

    return fwhm_m * 1e9

def print_theoretical_fwhm():
    print("\nTheoretical FWHM")
    print(" ")
    for order in [1, 2, 3]:
        fwhm_nm = calculate_theoretical_fwhm(order)
        print(
            f"Order {order}: "
            f"FWHM = {fwhm_nm:.3f} nm"
        )
def measure_fwhm_at_row(
    image,
    spatial_row,
    approximate_pixel,
    wavelength_model,
    search_half_width=FWHM_SEARCH_HALF_WIDTH
):
    """ Measure the FWHM of one isolated spectral line at one spatial row """
    
    left_index = max(
        0,
        approximate_pixel - search_half_width
    )

    right_index = min(
        image.shape[1],
        approximate_pixel + search_half_width + 1
    )

    local_profile = image[
        spatial_row,
        left_index:right_index
    ].copy()

    local_pixels = np.arange(
        left_index,
        right_index,
        dtype=np.float64
    )

    edge_values = np.concatenate(
        [
            local_profile[:4],
            local_profile[-4:]
        ]
    )

    background = np.median(edge_values)
    local_profile -= background
    local_peak_index = np.argmax(local_profile)
    peak_height = local_profile[local_peak_index]

    if peak_height <= 0:
        return np.nan
    (
        widths,
        width_heights,
        left_crossings,
        right_crossings
    ) = peak_widths(
        local_profile,
        [local_peak_index],
        rel_height=0.5
    )

    left_crossing_local = left_crossings[0]
    right_crossing_local = right_crossings[0]

    if (
        left_crossing_local <= 0
        or right_crossing_local >= len(local_profile) - 1
    ):
        return np.nan
    local_index_axis = np.arange(
        len(local_profile),
        dtype=np.float64
    )

    left_pixel = np.interp(
        left_crossing_local,
        local_index_axis,
        local_pixels
    )

    right_pixel = np.interp(
        right_crossing_local,
        local_index_axis,
        local_pixels
    )
    left_wavelength = wavelength_model(left_pixel)
    right_wavelength = wavelength_model(right_pixel)

    fwhm_nm = abs(
        right_wavelength - left_wavelength
    )

    return fwhm_nm

def measure_empirical_fwhm(
    corrected_images,
    wavelength_models
):
    """ Measure selected spectralline widths along the spatial axis"""

    wavelength_model = wavelength_models[SELECTED_POLYNOMIAL_DEGREE]["model"]

    results = {}

    print("\nEmpirical FWHM")
    print(" ")

    for line in FWHM_LINES:
        lamp = line["lamp"]
        approximate_pixel = line["approximate_pixel"]
        wavelength_nm = line["wavelength_nm"]

        image = corrected_images[lamp]

        fwhm_values = []

        for spatial_row in FWHM_SPATIAL_ROWS:
            fwhm_nm = measure_fwhm_at_row(
                image=image,
                spatial_row=spatial_row,
                approximate_pixel=approximate_pixel,
                wavelength_model=wavelength_model
            )

            fwhm_values.append(fwhm_nm)

        fwhm_values = np.asarray(
            fwhm_values,
            dtype=np.float64
        )

        key = f"{lamp}_{wavelength_nm:.2f}"

        results[key] = {
            "lamp": lamp,
            "wavelength_nm": wavelength_nm,
            "rows": FWHM_SPATIAL_ROWS.copy(),
            "fwhm_nm": fwhm_values,
        }

        valid_values = fwhm_values[
            np.isfinite(fwhm_values)
        ]

        print(
            f"{lamp}, {wavelength_nm:.2f} nm: "
            f"mean={np.mean(valid_values):.3f} nm, "
            f"median={np.median(valid_values):.3f} nm, "
            f"valid rows={len(valid_values)}"
        )

    return results

def plot_empirical_fwhm(fwhm_results):
    """ Plot measured FWHM against spatial pos """

    fig, ax = plt.subplots(
        figsize=(11, 7),
        constrained_layout=True
    )

    for result in fwhm_results.values():
        label = (
            f'{result["lamp"]}, '
            f'{result["wavelength_nm"]:.2f} nm'
        )

        ax.plot(
            result["rows"],
            result["fwhm_nm"],
            marker=".",
            markersize=4,
            linewidth=1,
            label=label
        )

    ax.set_xlabel("Spatial pixel")
    ax.set_ylabel("Measured FWHM [nm]")
    ax.set_title("Empirical spectral FWHM along the spatial axis")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path = RESULT_DIR / "empirical_fwhm.png"
    fig.savefig(output_path, dpi=200)

    return fig

def plot_calibrated_lamp_spectra(
    profiles,
    wavelength_models
):
    """
    Plot the lamp spectra against the calibrated wavelength axis """

    wavelength_model = wavelength_models[
        SELECTED_POLYNOMIAL_DEGREE]["model"]

    number_of_pixels = next(iter(profiles.values())).size
    spectral_pixels = np.arange(number_of_pixels)
    wavelengths_nm = wavelength_model(spectral_pixels)

    display_range = (
        (wavelengths_nm >= 400)
        & (wavelengths_nm <= 710)
    )

    fig, ax = plt.subplots(
        figsize=(13, 6),
        constrained_layout=True
    )

    lamp_colours = {
        "Argon": "tab:blue",
        "Mercury": "tab:orange",
        "CO2": "tab:green",
    }

    for name, profile in profiles.items():
        positive_profile = np.maximum(profile, 0)

        maximum = np.max(
            positive_profile[display_range]
        )

        if maximum <= 0:
            continue

        normalized_profile = (
            positive_profile / maximum
        )

        ax.plot(
            wavelengths_nm[display_range],
            normalized_profile[display_range],
            color=lamp_colours[name],
            linewidth=1.1,
            label=name
        )

    reference_lines = [
        ("Hg 404.7 nm", 404.7),
        ("Hg 435.8 nm", 435.8),
        ("Hg 546.1 nm", 546.1),
        ("Hg 577/579 nm", 578.02),
        ("Ar 696.5 nm", 696.54),
        ("Ar 706.7 nm", 706.72),
    ]

    for label, wavelength_nm in reference_lines:
        ax.axvline(
            wavelength_nm,
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=1
        )

        ax.text(
            wavelength_nm,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.20",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1
            }
        )

    ax.set_xlim(400, 710)
    ax.set_ylim(0, 1.18)

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized intensity [a.u.]")
    ax.set_title(
        "Calibrated spectra of Hg, Ar and CO$_2$ lamps"
    )

    ax.grid(
        color="0.85",
        linewidth=0.7,
        alpha=0.7
    )

    ax.legend(
        loc="upper right",
        frameon=True
    )

    output_path = (
        RESULT_DIR / "calibrated_lamp_spectra.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    return fig

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok = True)
    mean_dark, dark_paths = load_mean_dark_image(
        EXPOSURE_MS
    )

    print(f"\nExposure: {EXPOSURE_MS} ms")
    print(f"Number of dark images: {len(dark_paths)}")
    print(f"Dark-image shape: {mean_dark.shape}")
    print(f"Mean dark level: {mean_dark.mean():.3f}")
    print(f"Median dark level:{np.median(mean_dark):.3f}")

    raw_images = {}
    corrected_images = {}

    for name, path in LAMP_FILES.items():
        raw_image = load_image(path)

        if raw_image.shape != mean_dark.shape:
            raise ValueError(
                f"{path.name} has shape {raw_image.shape}, "
                f"but the dark image has shape {mean_dark.shape}"
            )

        corrected_image = raw_image - mean_dark

        raw_images[name] = raw_image
        corrected_images[name] = corrected_image

        print_img_info(
            name,
            raw_image,
            corrected_image
        )

    profiles = {
        name: extract_centre_profile(image)
        for name, image in corrected_images.items()
    }

    plot_corrected_images(corrected_images)
    plot_centre_profiles(profiles)
    plot_combined_image(corrected_images)

    peak_figure, detected_peaks = detect_peaks_in_profiles(
        profiles
    )

    pixels, wavelengths, wavelength_models = (
    fit_wavelength_models(CALIBRATION_POINTS)
    )

    plot_wavelength_models(
        CALIBRATION_POINTS,
        pixels,
        wavelengths,
        wavelength_models
    )

    plot_calibration_rmse(wavelength_models)
    print_theoretical_fwhm()

    fwhm_results = measure_empirical_fwhm(corrected_images,wavelength_models)
    plot_empirical_fwhm(fwhm_results)

    plot_calibrated_lamp_spectra(profiles,wavelength_models)

    print(f"\nFigures saved in: {RESULT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
    

