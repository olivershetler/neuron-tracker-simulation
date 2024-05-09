import yaml
from pathlib import Path
import os


def load_env_config():
    env_dict = {}
    for key, value in os.environ.items():
        env_dict[key] = value
    return env_dict

def load_config():
    config_dict = {}
    paths = Path(__file__).parent.glob('*.yaml')
    for path in paths:
        section = path.stem
        if section == "env":
            raise ValueError("The filename env.yaml is a reserved file name. Rename that config file file.")
        with open(path, 'r') as f:
            config_dict[section] = yaml.safe_load(f)
    config_dict["env"] = load_env_config()
    return config_dict