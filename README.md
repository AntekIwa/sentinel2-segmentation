# Sentinel-2 Segmentation

This project explores unsupervised analysis of Sentinel-2 multispectral imagery with Principal Component Analysis (PCA), Spectral Angle Mapper (SAM), and K-Means clustering. It includes experiments for Almeria segmentation, the Sahara Eye, and burn scar detection on Evia, Greece.

## Project Layout

```text
.
|-- data/                  # Sentinel-2 input bands grouped by study area
|-- notebooks/             # Jupyter notebooks used for exploration and reports
|-- outputs/               # Generated figures and raster outputs
|   |-- almeria/
|   |-- fire_evia/
|   `-- sahara_eye/
|-- reports/               # PDF and LaTeX project reports
|-- scripts/               # Re-runnable analysis scripts
|-- data_utils.py          # Raster loading, resampling, and image helpers
|-- pca_utils.py           # Custom PCA/SVD implementation
|-- requirements.txt
`-- README.md
```

## Setup

Create a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The main dependencies are `numpy`, `matplotlib`, `rasterio`, `scikit-learn`, and Jupyter.

## Data

The repository expects Sentinel-2 bands under `data/`, with one folder per area. The scripts currently use:

- `data/Sahara_oko/` for the Sahara Eye SAM example.
- `data/grecja/Po_pozarze/` for the Evia burn scar PC3 example.
- `data/almeria/` for the Almeria notebook and report experiments.

Band filenames follow the EO Browser export pattern, for example:

```text
2026-04-30-00_00_2026-04-30-23_59_Sentinel-2_L2A_B04_(Raw).tiff
```

## Running Scripts

Run scripts from the repository root:

```powershell
python scripts\sahara_eye_detection.py
python scripts\fire_evia_pca3.py
```

Both scripts read from `data/` and save new figures into `outputs/`.

## Notebooks and Reports

The exploratory notebooks are in `notebooks/`. Start Jupyter from the repository root so imports such as `from pca_utils import ...` resolve correctly:

```powershell
jupyter lab
```

The main LaTeX/PDF reports are stored in `reports/`. They use lightweight report figures from `reports/figures/pl/` and `reports/figures/en/`, with the full generated outputs kept in `outputs/almeria/`.

## Notes

- Large Sentinel-2 raster files are kept under `data/`.
- Generated figures and segmentation rasters are grouped under `outputs/`.
- Local IDE files, virtual environments, Python caches, notebook checkpoints, and LaTeX temporary files are ignored by Git.
