from dagster import ConfigurableResource
from pathlib import Path

class DataDirectory(ConfigurableResource):
    path: str

    def subdir(self, name: str) -> Path:
        subdir = Path(self.path).resolve() / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir