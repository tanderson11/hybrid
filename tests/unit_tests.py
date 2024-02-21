import unittest
import re
import os
import numpy as np
import pandas as pd
import glob
import pathlib
from dataclasses import dataclass
from typing import NamedTuple

import reactionmodel.load
from hybrid.simulators import SIMULATORS
import hybrid.hybrid as hybrid

# wherever we are, save test output to test_output folder
test_out = './test_output/'

def get_path_from_check_string(check_string, prefix, directory_name, filename):
    # we have a check string like p01m01i01 for parameters 1, model 1, and initial condition 1
    # this function takes prefix (e.g. p/m/i) and finds the txt file that specifies the desired configuration
    match = re.search(f'{prefix}([0-9]+)', check_string)
    model_path = f'{directory_name}/{filename}{match[1]}.txt' if match is not None else f'{filename}.txt'
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

    test_results = {}
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

def extend_with_tests_from_dir(tests, dir):
    # crawl dir to find all tests in that dir and add them to the list `tests` passed as an argument
    print(f"Extending test suite from {dir}\nAll directories within {os.path.join(dir, 'checks/')} that match specifications in {dir} will be used as tests.")
    model_paths  = get_files(dir, 'model.txt', 'models', 'model*.txt')
    params_paths = get_files(dir, 'parameters.txt', 'parameters', 'parameters*.txt')
    config_paths = get_files(dir, 'config.txt', 'configurations', 'config*.txt')
    ic_paths     = get_files(dir, 'ic.txt', 'initial_conditions', 'initial*.txt')
    specifications = {}
    for model_path in model_paths:
        for params_path in params_paths:
            for config_path in config_paths:
                for ic_path in ic_paths:
                    specification = reactionmodel.load.load_specification(model_path, params_path, config_path, ic_path)
                    # use the parameter and ic file names as a unique identifier for this combination
                    # later, we will look up all the combinations that we have test data for, and run simulations to check
                    model_match = re.search('[a-z]+([0-9]+)\.txt', model_path)
                    config_match = re.search('[a-z]+([0-9]+)\.txt', config_path)
                    param_match = re.search('[a-z]+([0-9]+)\.txt', params_path)
                    ic_match = re.search('[a-z]+([0-9]+)\.txt', ic_path)
                    matches = [('m', model_match), ('c', config_match), ('p', param_match), ('i', ic_match)]
                    identifier = ''
                    for id_str, match in matches:
                        if match:
                            identifier += id_str + str(match[1])
                    # if identifier == '': all of the configuration files lived in root directory, so the check should just live in the root of the check directory
                    specifications[identifier] = specification
    return extend_via_checks(tests, dir, specifications)

sbml_tests = []
sbml_root = os.path.join(os.path.dirname(__file__), "sbml-tests/")
test_dirs = glob.glob(os.path.join(sbml_root, 'sbml-*'))
for t_dir in test_dirs:
    extend_with_tests_from_dir(sbml_tests, t_dir)

@dataclass
class SimulatorArguments():
    t_span: tuple
    t_eval: tuple

class TestSBMLMeta(type):
    def __new__(mcs, names, bases, dct):
        def gen_test(test_name, specification, check_file):
            def test(self):
                self.test_name = test_name
                self.specification = specification
                self.check_file = check_file
                # load the csv of analytic/high quality simulation results
                self.check_data = pd.read_csv(self.check_file)
                print("About to run test.")
                self._test_single()
            return test

        for root, spec_name, specification, check_file in sbml_tests:
            test_name = f'{os.path.basename(os.path.normpath(root))}_{spec_name}'
            dct[f'test_{test_name}'] = gen_test(test_name, specification, check_file)
        return type.__new__(mcs, names, bases, dct)

class TestSBML(unittest.TestCase, metaclass=TestSBMLMeta):
    TEST_ARGUMENTS = SimulatorArguments((0.0, 50.0), np.linspace(0, 50, 51))
    n = 10

    class TestResult(NamedTuple):
        results_df: pd.DataFrame
        check_df: pd.DataFrame
        z_scores_for_mean_by_species: pd.DataFrame

    def tearDown(self):
        # save results
        results_table = self.test_result.results_df
        z_ts = self.test_result.z_scores_for_mean_by_species
        out = os.path.join(test_out, self.test_name)
        pathlib.Path(out).mkdir(parents=True, exist_ok=True) 
        results_table.to_csv(os.path.join(out, f'n={self.n}_simulation_results.csv'))
        z_ts.to_csv(os.path.join(out, f'n={self.n}_simulation_zscores.csv'))

    def _test_single(self):
        desired_species = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        all_species = [s.name for s in self.specification.model.species]
        targets = [all_species.index(s) for s in desired_species]

        results = self.do_simulations(targets, desired_species)

        df = pd.concat(aligned_results, axis=1)
        all_results = pd.concat([df.groupby(by=df.columns, axis=1).mean(), df.groupby(by=df.columns, axis=1).std()], axis=1)

        check_targets = set([c.split('-')[0] for c in check_data.columns if len(c.split('-')) > 1])
        all_results.columns = [c + '-mean' if i < len(check_targets) else c + '-sd' for i,c in enumerate(results_to_check.columns)]

        z_ts = self.z_score_for_mean(all_results, check_targets, self.check_data, self.n)

        self.test_result = self.TestResult(all_results, self.check_data, z_ts)

        # assert something about zscores
        # TK

    def do_simulations(self, targets, desired_species):
        aligned_results = []
        simulator = self.specification.simulator
        forward_time = SIMULATORS[simulator]
        rng = np.random.default_rng()
        initial_condition = self.specification.model.make_initial_condition(self.specification.initial_condition)
        simulation_options = self.specification.simulation_options.copy()

        if simulator == 'hybrid':
            partition_path = simulation_options.pop('partition')
            partition_scheme = hybrid.load_partition_scheme(partition_path)
            simulation_options['partition_function'] = partition_scheme.partition_function
        k = self.specification.model.get_k(parameters=self.specification.parameters, jit=True)
        for i in range(self.n):
            print(i)
            result = forward_time(initial_condition, self.TEST_ARGUMENTS.t_span, k, self.specification.model.stoichiometry(), self.specification.model.rate_involvement(), rng, discontinuities=self.TEST_ARGUMENTS.t_eval, **simulation_options)
            self.align_single_result(result, self.check_data['time'], targets, desired_species)
            #        aligned = self.align_results(results, self.check_data['time'], targets, desired_species)
            aligned_results.append(result)
        return aligned_results

    @staticmethod
    def align_single_result(r, time, target_indices, species_names):
        aligned = []
        t_history = r.t_history
        for t in time:
            idx = np.argmin(np.abs(t-t_history))
            aligned.append((r.t_history[idx], *[r.y_history[target_index,idx] for target_index in target_indices]))
        indexed_results = pd.DataFrame.from_records(aligned, columns=['time', *species_names])
        indexed_results['time'] = np.round(indexed_results['time'], 5)
        indexed_results.set_index('time')

        return indexed_results

    @staticmethod
    def z_score_for_mean(all_results, target_species, check_data, n):
        # https://github.com/sbmlteam/sbml-test-suite/blob/release/cases/stochastic/DSMTS-userguide-31v2.pdf
        z_ts = {}
        for species in target_species:
            z_t = (all_results[f'{species}-mean'] - check_data[f'{species}-mean'])/(check_data[f'{species}-sd']) * np.sqrt(n)
            z_ts[species] = z_t
        
        z_ts = pd.DataFrame(z_ts)

        return z_ts

if __name__ == '__main__':
    unittest.main()