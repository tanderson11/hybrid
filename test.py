import re
import os
from typing import NamedTuple

def get_path_from_check_string(check_string, prefix, directory_name, filename):
    match = re.search(f'{prefix}([0-9]+)', check_string)
    model_path = f'{directory_name}/{filename}{match[1]}.txt' if match is not None else f'{filename}.txt'
    return model_path

class SpecificationPaths(NamedTuple):
    model_path: str
    parameters_path: str
    config_path: str
    initial_condition_path: str

def get_path_tuple(root, check):
    model_path = os.path.join(root, get_path_from_check_string(check, 'm', 'models', 'model'))
    parameters_path = os.path.join(root, get_path_from_check_string(check, 'p', 'parameters', 'parameters'))
    config_path = os.path.join(root, get_path_from_check_string(check, 'c', 'configs', 'config'))
    ic_path = os.path.join(root, get_path_from_check_string(check, 'i', 'initial_conditions', 'initial'))

    return SpecificationPaths(model_path, parameters_path, config_path, ic_path)