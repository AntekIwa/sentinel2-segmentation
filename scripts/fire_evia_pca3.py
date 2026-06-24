from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import (  # noqa: E402
    przywroc_format_obrazka,
    rozciagnij_na_2d,
    stworz_macierz_3d_z_resamplingiem,
)
from pca_utils import pca_z_autorskim_svd  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "grecja" / "Po_pozarze"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fire_evia"
PREFIX = "2021-08-18-00_00_2021-08-18-23_59_Sentinel-2_L2A_"
SUFFIX = "_(Raw).tiff"


def band_path(band):
    path = DATA_DIR / f"{PREFIX}{band}{SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"Missing Sentinel-2 band: {path}")
    return str(path)


paths_10m = [band_path(band) for band in ("B02", "B03", "B04", "B08")]
paths_20m = [band_path(band) for band in ("B11", "B12")]

print("-> Loading data and computing PCA...")
cube = stworz_macierz_3d_z_resamplingiem(paths_10m, paths_20m)
table_2d, original_shape = rozciagnij_na_2d(cube)
pca_result_2d, _, _ = pca_z_autorskim_svd(table_2d, liczba_skladowych=3)
pca_image_3d = przywroc_format_obrazka(pca_result_2d, original_shape)

pca3_raw = pca_image_3d[:, :, 2]
vmin, vmax = np.percentile(pca3_raw, [2, 98])

print("-> Rendering PC3 burn scar map...")
fig, ax = plt.subplots(figsize=(12, 10))
image = ax.imshow(pca3_raw, cmap="inferno", vmin=vmin, vmax=vmax)

ax.set_title(
    "PC3 Component - Burn Scar Separation\nEvia, Greece, 18 Aug 2021",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax.axis("off")

fig.colorbar(
    image,
    ax=ax,
    orientation="horizontal",
    pad=0.04,
    label="PC3 anomaly intensity",
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "fire_evia_pc3.png"
fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")

print(f"=== DONE: saved {output_path} ===")
plt.tight_layout()
plt.show()
