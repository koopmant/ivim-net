from pathlib import Path
from shutil import copyfile
from ruamel.yaml import YAML


def read_yaml(yaml_file_path: Path):
    if not yaml_file_path.exists():
        copyfile(str(yaml_file_path.with_suffix('.yaml.example')), yaml_file_path)

    yaml = YAML(typ='safe')
    return yaml.load(yaml_file_path)
