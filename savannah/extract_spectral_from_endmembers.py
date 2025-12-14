from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np


# -------------------------------
# Configuration (GitHub-safe)
# -------------------------------

BASE_DIR = Path.home() / "thesis_data"
POINTS_DIR = BASE_DIR / "Pure_points"
OUTPUT_DIR = BASE_DIR / "Pure_spectral"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Function
# -------------------------------

def get_pure_spectral(image_path: Path, point_name: str, image_type: str) -> None:
    """
    Sample raster spectral values at point locations and save to CSV.

    Parameters
    ----------
    image_path : Path
        Path to the raster image.
    point_name : str
        Name of the point file (without extension).
    image_type : str
        Identifier for the image type (e.g., 'sentinel2', 'enmap').
    """

    # Load point data (GeoPackage)
    point_file = POINTS_DIR / f"{point_name}.gpkg"
    gdf = gpd.read_file(point_file)

    # Extract coordinates (ensure CRS matches raster CRS)
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    # Sample raster at point locations
    with rasterio.open(image_path) as src:
        samples = list(src.sample(coords))

    # Convert to NumPy array (rows = points, columns = bands)
    samples = np.asarray(samples)

    # Save to CSV
    output_file = OUTPUT_DIR / f"{point_name}_{image_type}.csv"
    np.savetxt(output_file, samples, delimiter=",", fmt="%i")

    print(f"Saved {point_name} spectra ({image_type}) → {output_file}")


# -------------------------------
# Script entry point
# -------------------------------

if __name__ == "__main__":

    SENTINEL2_IMAGE = BASE_DIR / "sentinel_stm_dry.tif"
    ENMAP_IMAGE = BASE_DIR / "20230502_SPECTRAL_IMAGE.TIF"

    CLASS_NAMES = ["woody", "herbaceous", "soil"]

    for cls in CLASS_NAMES:
        get_pure_spectral(SENTINEL2_IMAGE, cls, "sentinel2")
        get_pure_spectral(ENMAP_IMAGE, cls, "enmap")
