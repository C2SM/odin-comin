from zipfile import ZipFile
from urllib.request import urlopen
import geopandas as gpd
from pathlib import Path
from io import BytesIO

def get_natural_earth(
    plotdir,
    resolution: str = "10m",
    category: str = "physical",
    name: str = "coastline",
) -> gpd.GeoDataFrame:
    """Download the natural earth file and returns it.

    For more information about the natural earth data, see
    `the natural earth website <https://www.naturalearthdata.com/>`_.

    As this function reads a large dataset, it caches it in case of many uses
    (ex: testing).

    :arg resolution: The resolution for the used shapefile.
        Available resolutions are: '10m', '50m', '110m'
    :arg category: The category of the shapefile.
        Available categories are: 'physical', 'cultural', 'raster'
    :arg name: The name of the shapefile.
        Category of the data to download. Many are availables, look at the
        natural earth website for more info.

    :returns: The shapefile as a GeoDataFrame

    """
    path_to_save = Path(plotdir) / "natural_earth" / f"ne_{resolution}_{category}_{name}"
    if not path_to_save.exists():
        URL_TEMPLATE = (
            "https://naturalearth.s3.amazonaws.com/{resolution}_"
            "{category}/ne_{resolution}_{name}.zip"
        )
        path_to_save.mkdir(parents=True, exist_ok=True)
        url = URL_TEMPLATE.format(resolution=resolution, category=category, name=name)
        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(path_to_save)

    # Load the country file
    shpfile = str(path_to_save / f"ne_{resolution}_{name}.shp")
    return gpd.read_file(shpfile)