import re
import os
import glob
from typing import NamedTuple

import reactionmodel.parser

from hybrid.parse import PreconfiguredSimulatorLoader

def discover_tests(root, test_directory_pattern='*'):
    tests = []
    test_dirs = glob.glob(os.path.join(root, test_directory_pattern))
    for t_dir in test_dirs:
        extend_with_tests_from_dir(tests, t_dir)
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

def extend_via_checks(tests, root, specifications):
    # use the subdirectories of root/checks to determine what tests exist
    tests_to_do = []
    for check in glob.glob(os.path.join(root, 'checks', '*/')):
        check_dir_root = check.split('/')[-2]
        tests_to_do.append((check_dir_root, specifications[check_dir_root], check))

    for spec_name, specification, check_dir in tests_to_do:
        # each check directory contains 1 CSV file with a name like check{SBML_TEST_NUMBER}.csv
        assert(len(glob.glob(os.path.join(check_dir, 'check*.csv')))) == 1, f"Check directory {check_dir} had more than 1 check csv. I don't know what to do"
        check_file = glob.glob(os.path.join(check_dir, 'check*.csv'))[0]
        tests.append((root, spec_name, specification, check_file))
    return tests

def get_files(root, individual, collection, pattern):
    if os.path.isfile(os.path.join(root, individual)):
        return [os.path.join(root, individual)]
    return glob.glob(os.path.join(root, collection, pattern))

def load_specification(model_path, params_path, config_path, ic_path):
    model = reactionmodel.parser.load(model_path).model
    parameters = reactionmodel.parser.load(params_path).parameters
    simulation_config = reactionmodel.parser.load(config_path, ConfigParser=PreconfiguredSimulatorLoader).simulator_config
    initial_condition = reactionmodel.parser.load(ic_path).initial_condition

    return reactionmodel.parser.ParseResults(model, parameters, initial_condition, simulation_config)

def extend_with_tests_from_dir(tests, dir):
    # crawl dir to find all tests in that dir and add them to the list `tests` passed as an argument
    print(f"Extending test suite from {dir}\nAll directories within {os.path.join(dir, 'checks/')} that match specifications in {dir} will be used as tests.")
    model_paths  = get_files(dir, 'model.yaml', 'models', 'model*.yaml')
    params_paths = get_files(dir, 'parameters.yaml', 'parameters', 'parameters*.yaml')
    config_paths = get_files(dir, 'config.yaml', 'configurations', 'config*.yaml')
    ic_paths     = get_files(dir, 'ic.yaml', 'initial_conditions', 'initial*.yaml')
    specifications = {}
    for model_path in model_paths:
        for params_path in params_paths:
            for config_path in config_paths:
                for ic_path in ic_paths:
                    specification = load_specification(model_path, params_path, config_path, ic_path)
                    # use the parameter and ic file names as a unique identifier for this combination
                    # later, we will look up all the combinations that we have test data for, and run simulations to check
                    model_match = re.search('[a-z]+([0-9]+)\.yaml', model_path)
                    config_match = re.search('[a-z]+([0-9]+)\.yaml', config_path)
                    param_match = re.search('[a-z]+([0-9]+)\.yaml', params_path)
                    ic_match = re.search('[a-z]+([0-9]+)\.yaml', ic_path)
                    matches = [('m', model_match), ('c', config_match), ('p', param_match), ('i', ic_match)]
                    identifier = ''
                    for id_str, match in matches:
                        if match:
                            identifier += id_str + str(match[1])
                    # if identifier == '': all of the configuration files lived in root directory, so the check should just live in the root of the check directory
                    specifications[identifier] = specification
    return extend_via_checks(tests, dir, specifications)