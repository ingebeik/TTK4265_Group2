from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
RESULT_DIR = PROJECT_DIR / "results" / "radiometric"

EXPOSURE_TIMES_MS = [200, 300, 400]
NUMBER_OF_IMAGES = 10

SATURATION_VALUE = 65535
NEAR_SATURATION_VALUE = 0.99 * SATURATION_VALUE




def load_image(path):
    """Load one two-dimensional grayscale image."""

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find image: {path}"
        )

    image = np.asarray(
        Image.open(path),
        dtype=np.float64
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional grayscale image, "
            f"but {path.name} has shape {image.shape}"
        )

    return image


def load_image_stack(prefix, exposure_ms):
    """
    Load the ten R or D images for one exposure time.

    For example, prefix='R' and exposure_ms=300 loads
    R300.1.png through R300.10.png.
    """

    paths = [
        IMAGE_DIR / f"{prefix}{exposure_ms}.{index}.png"
        for index in range(1, NUMBER_OF_IMAGES + 1)
    ]

    images = [
        load_image(path)
        for path in paths
    ]

    expected_shape = images[0].shape

    for path, image in zip(paths, images):
        if image.shape != expected_shape:
            raise ValueError(
                f"{path.name} has shape {image.shape}, "
                f"expected {expected_shape}"
            )

    stack = np.stack(
        images,
        axis=0
    )

    return stack, paths


def print_saturated_coordinates(
    saturation_mask,
    exposure_ms
):
    """Print locations that reached the 16-bit saturation value."""

    saturated_coordinates = np.argwhere(
        saturation_mask
    )

    if saturated_coordinates.size == 0:
        print("Saturated R-pixel locations: none")
        return

    # Remove frame index, retaining spatial row and spectral column.
    spatial_spectral_coordinates = (
        saturated_coordinates[:, 1:3]
    )

    unique_coordinates, occurrence_counts = np.unique(
        spatial_spectral_coordinates,
        axis=0,
        return_counts=True
    )

    print(
        f"Saturated R-pixel locations at "
        f"{exposure_ms} ms:"
    )

    for coordinate, count in zip(
        unique_coordinates,
        occurrence_counts
    ):
        spatial_row = coordinate[0]
        spectral_pixel = coordinate[1]

        print(
            f"  row {spatial_row}, "
            f"spectral pixel {spectral_pixel}: "
            f"saturated in {count}/"
            f"{NUMBER_OF_IMAGES} frames"
        )


def process_exposure(exposure_ms):
    """Average and dark-correct one exposure series."""

    radiometric_stack, radiometric_paths = (
        load_image_stack(
            prefix="R",
            exposure_ms=exposure_ms
        )
    )

    dark_stack, dark_paths = (
        load_image_stack(
            prefix="D",
            exposure_ms=exposure_ms
        )
    )

    if radiometric_stack.shape != dark_stack.shape:
        raise ValueError(
            f"R and D stacks at {exposure_ms} ms "
            f"have different shapes: "
            f"{radiometric_stack.shape} and "
            f"{dark_stack.shape}"
        )

    mean_radiometric = np.mean(
        radiometric_stack,
        axis=0
    )

    mean_dark = np.mean(
        dark_stack,
        axis=0
    )

    corrected_signal = (
        mean_radiometric - mean_dark
    )

    exposure_seconds = exposure_ms / 1000.0

    signal_rate = (
        corrected_signal / exposure_seconds
    )

    # Check saturation only in the illuminated R images.
    saturation_mask = (
        radiometric_stack >= SATURATION_VALUE
    )

    near_saturation_mask = (
        radiometric_stack >= NEAR_SATURATION_VALUE
    )

    saturated_pixels_per_frame = np.count_nonzero(
        saturation_mask,
        axis=(1, 2)
    )

    near_saturated_pixels_per_frame = np.count_nonzero(
        near_saturation_mask,
        axis=(1, 2)
    )

    print(f"\nExposure: {exposure_ms} ms")
    print("-" * 30)

    print(
        f"R images: {len(radiometric_paths)}"
    )

    print(
        f"D images: {len(dark_paths)}"
    )

    print(
        f"Image shape: {mean_radiometric.shape}"
    )

    print(
        f"Mean dark level: "
        f"{mean_dark.mean():.3f}"
    )

    print(
        f"Median dark level: "
        f"{np.median(mean_dark):.3f}"
    )

    print(
        f"Maximum R value: "
        f"{radiometric_stack.max():.1f}"
    )

    print(
        f"Maximum corrected mean: "
        f"{corrected_signal.max():.1f}"
    )

    print(
        f"Exact saturated pixels per frame: "
        f"{saturated_pixels_per_frame.tolist()}"
    )

    print(
        f"Pixels above 99% per frame: "
        f"{near_saturated_pixels_per_frame.tolist()}"
    )

    print_saturated_coordinates(
        saturation_mask=saturation_mask,
        exposure_ms=exposure_ms
    )

    return {
        "exposure_ms": exposure_ms,
        "exposure_seconds": exposure_seconds,
        "radiometric_stack": radiometric_stack,
        "dark_stack": dark_stack,
        "mean_radiometric": mean_radiometric,
        "mean_dark": mean_dark,
        "corrected_signal": corrected_signal,
        "signal_rate": signal_rate,
        "saturation_mask": saturation_mask,
        "saturated_pixels_per_frame": (
            saturated_pixels_per_frame
        ),
        "near_saturated_pixels_per_frame": (
            near_saturated_pixels_per_frame
        ),
    }


def show_radiometric_image_for_roi(results):
    """
    Show the mean 300 ms dark-corrected image.

    This is a diagnostic figure used to select the illuminated
    calibration-screen region. It is not saved as a report figure.
    """

    exposure_ms = 300

    image = results[
        exposure_ms
    ]["corrected_signal"]

    display_image = np.maximum(
        image,
        0
    )

    display_maximum = np.percentile(
        display_image,
        99.5
    )

    fig, ax = plt.subplots(
        figsize=(13, 6),
        constrained_layout=True
    )

    shown_image = ax.imshow(
        display_image,
        cmap="inferno",
        aspect="auto",
        vmin=0,
        vmax=display_maximum
    )

    ax.set_title(
        "Mean dark-corrected radiometric image, "
        f"{exposure_ms} ms"
    )

    ax.set_xlabel("Spectral pixel")
    ax.set_ylabel("Spatial pixel")

    fig.colorbar(
        shown_image,
        ax=ax,
        label="Dark-corrected counts"
    )

    return fig


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for exposure_ms in EXPOSURE_TIMES_MS:
        results[exposure_ms] = process_exposure(
            exposure_ms
        )

    print(
        "\nAll radiometric and dark images "
        "were loaded successfully."
    )

    show_radiometric_image_for_roi(
        results
    )

    plt.show()


if __name__ == "__main__":
    main()