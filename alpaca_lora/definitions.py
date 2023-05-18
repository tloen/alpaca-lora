from dagster import Definitions, load_assets_from_modules
from . import assets
from .resources import DataDirectory

definitions = Definitions(
    assets=load_assets_from_modules([assets]),
    resources={"data_dir": DataDirectory(path="./data")}
)