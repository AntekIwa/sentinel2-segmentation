from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import (  # noqa: E402
    kat_widmowy,
    przywroc_format_obrazka,
    rozciagnij_na_2d,
    stworz_macierz_3d_z_resamplingiem,
)
from pca_utils import pca_z_autorskim_svd  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "Sahara_oko"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sahara_eye"
PREFIX = "2026-04-30-00_00_2026-04-30-23_59_Sentinel-2_L2A_"
SUFFIX = "_(Raw).tiff"


def band_path(band):
    path = DATA_DIR / f"{PREFIX}{band}{SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"Missing Sentinel-2 band: {path}")
    return str(path)


paths_10m = [band_path(band) for band in ("B02", "B03", "B04", "B08")]
paths_20m = [band_path(band) for band in ("B05", "B06", "B07", "B8A", "B11", "B12")]

print("=== STEP 1: Sahara Eye spatial analysis ===")
cube = stworz_macierz_3d_z_resamplingiem(paths_10m, paths_20m)

raw_rgb = cube[:, :, [2, 1, 0]].astype(float)
brightness_max = np.percentile(raw_rgb, 99.5)
if brightness_max == 0:
    brightness_max = 1.0
rgb_preview = np.clip(raw_rgb, 0, brightness_max) / brightness_max

table_2d, original_shape = rozciagnij_na_2d(cube)
pca_result_2d, _, _ = pca_z_autorskim_svd(table_2d, liczba_skladowych=3)
pca_image_3d = przywroc_format_obrazka(pca_result_2d, original_shape)

print("\n=== STEP 2: Sampling rock signature ===")
target_lon = -11.550
target_lat = 21.100

with rasterio.open(paths_10m[0]) as dataset:
    pixel_y, pixel_x = dataset.index(target_lon, target_lat)

sahara_signature = pca_image_3d[pixel_y, pixel_x, :]
print(f"-> Signature sampled at pixel X:{pixel_x}, Y:{pixel_y}")

print("\n=== STEP 3: Spectral classification ===")
raw_angles = kat_widmowy(pca_image_3d, sahara_signature)

angle_threshold = 0.20
similarity_map = (angle_threshold - np.clip(raw_angles, 0, angle_threshold)) / angle_threshold

print("\n-> Rendering result...")
fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(rgb_preview)

masked_similarity = np.ma.masked_where(similarity_map < 0.1, similarity_map)
image = ax.imshow(masked_similarity, cmap="inferno", alpha=0.85)

colorbar = fig.colorbar(image, ax=ax, shrink=0.7)
colorbar.set_label("Rock signature similarity (SAM)", fontsize=12)

ax.set_title(
    "Spectral Detection of the Sahara Eye Rock Formation",
    fontsize=15,
    fontweight="bold",
)
ax.axis("off")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "sahara_eye_sam_detection.png"
fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")

print(f"=== DONE: saved {output_path} ===")
plt.show()
