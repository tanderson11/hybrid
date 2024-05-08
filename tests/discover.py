import re
import os
import glob
from typing import NamedTuple
from itertools import product

import reactionmodel.parser

from hybrid.parse import PreconfiguredSimulatorLoader

def discover_tests(root, test_directory_pattern='*', include_check=False, **spec_kwargs):
    tests = []
    test_dirs = glob.glob(os.path.join(root, test_directory_pattern))
    for t_dir in test_dirs:
        if not os.path.isdir(t_dir):
            continue
        extend_with_tests_from_dir(tests, t_dir, include_check=include_check, **spec_kwargs)
    return tests

def get_path_from_check_string(check_string, prefix, directory_name, filename):
    # we have a check string like p01m01i01 for parameters 1, model 1, and initial condition 1
    # this function takes prefix (e.g. p/m/i) and finds the yaml file that specifies the desired configuration
    match = re.search(f'{prefix}([0-9]+)', check_string)
    model_path = f'{directory_name}/{filename}{match[1]}.yaml' if match is not None else f'{filename}.yaml'
    return model_path

class SpecificationPaths(NamedTuple):
    model_path: str
    parameters_path: str
    config_path: str
    initial_condition_path: str

def get_path_tuple(root, check):
    # for each component of a specification, find the file specifying that component
    # return the files grouped as a tuple
    model_path = os.path.join(root, get_path_from_check_string(check, 'm', 'models', 'model'))
    parameters_path = os.path.join(root, get_path_from_check_string(check, 'p', 'parameters', 'parameters'))
    config_path = os.path.join(root, get_path_from_check_string(check, 'c', 'configs', 'config'))
    ic_path = os.path.join(root, get_path_from_check_string(check, 'i', 'initial_conditions', 'initial'))

    return SpecificationPaths(model_path, parameters_path, config_path, ic_path)

def get_files(root, individual, collection, pattern):
    if os.path.isfile(os.path.join(root, individual)):
        return [os.path.join(root, individual)]
    return glob.glob(os.path.join(root, collection, pattern))

def specs_from_dir(dir, format='yaml', model_base = 'model', params_base = 'parameters', config_base = 'simulator', t_base='t', initial_base = 'initial', simulators_share_checks=False):
    model_paths  = get_files(dir, f'{model_base}.{format}', 'models', f'*.{format}')
    params_paths = get_files(dir, f'{params_base}.{format}', 'parameters', f'*.{format}')
    config_paths = get_files(dir, f'{config_base}.{format}', 'simulators', f'*.{format}')
    t_paths      = get_files(dir,  f'{t_base}.{format}', 'ts', f'*.{format}')
    ic_paths     = get_files(dir, f'{initial_base}.{format}', 'initial_conditions', f'*.{format}')
    specifications = {}
    for model_path, params_path, config_path, t_path, ic_path in product(model_paths, params_paths, config_paths, t_paths, ic_paths):
        specification = reactionmodel.parser.load(model_path, params_path, config_path, t_path, ic_path, format=format, ConfigParser=PreconfiguredSimulatorLoader)
        # use the parameter and ic file names as a unique identifier for this combination
        # later, we will look up all the combinations that we have test data for, and run simulations to check
        model = os.path.basename(model_path).split('.')[0]
        model = model if model != model_base else None

        params = os.path.basename(params_path).split('.')[0]
        params = params if params != params_base else None

        simulator = os.path.basename(config_path).split('.')[0]
        simulator = simulator if simulator != config_base else None

        t = os.path.basename(t_path).split('.')[0]
        t = t if t != t_base else None

        initial = os.path.basename(ic_path).split('.')[0]
        initial = initial if initial != initial_base else None

        if simulators_share_checks:
            names = [model, params, t, initial]
        else:
            [model, params, t, initial, simulator]
        names = [n for n in names if n is not None]
        identifier = ''
        for i, name in enumerate(names):
            if i != 0:
                identifier += '_'
            identifier += name
        # if identifier == '': all of the configuration files lived in root directory, so the check should just live in the root of the check directory
        specifications[identifier] = specification
    return specifications

def specs_to_tests(root, specs, check_container='checks', include_check=False):
    tests = []
    for spec_name, specification in specs.items():
        check_file = None
        # if we're including checks, skip if the specification permutation does not have a check folder
        if include_check:
            check_dir = os.path.join(root, check_container, spec_name)
            if (not os.path.exists(check_dir)) or (not os.path.isdir(check_dir)):
                continue
            n_checks = len(glob.glob(os.path.join(check_dir, '*.csv')))
            assert n_checks <= 1, f"Check directory {check_dir} had more than 1 check csv. I don't know what to do"
            if n_checks == 1:
                check_file = glob.glob(os.path.join(check_dir, '*.csv'))[0]

        tests.append((root, spec_name, specification, check_file))

    return tests

def extend_with_tests_from_dir(tests, dir, include_check=False, **spec_kwargs):
    # crawl dir to find all tests in that dir and add them to the list `tests` passed as an argument
    print(f"Extending test suite from {dir}\n")
    new_specs = specs_from_dir(dir, **spec_kwargs)
    new_tests = specs_to_tests(dir, new_specs, include_check=include_check)
    return tests.extend(new_tests)