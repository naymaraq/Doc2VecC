import yaml


def load_cfg(yaml_file_path: str):
    with open(yaml_file_path, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    return cfg
