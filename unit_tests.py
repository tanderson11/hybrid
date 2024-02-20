import unittest
import re
import os
import numpy as np
import pandas as pd
import glob
from dataclasses import dataclass
from typing import NamedTuple
import reactionmodel.load
from simulators import SIMULATORS

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

def extend_via_checks(tests, root, specifications):
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
sbml_root = "./tests/sbml-tests/"
test_dirs = glob.glob(os.path.join(sbml_root, 'sbml-*'))
for t_dir in test_dirs:
    extend_with_tests_from_dir(sbml_tests, t_dir)

@dataclass
class SimulatorArguments():
    t_span: tuple
    t_eval: tuple

class TestFromSpecificationMeta(type):
    def __new__(mcs, names, bases, dct):
        def gen_test(specification, check_file):
            def test(self):
                self.specification = specification
                self.check_file = check_file
                # load the csv of analytic/high quality simulation results
                self.check_data = pd.read_csv(self.check_file)

                self._test_single()
            return test

        for root, spec_name, specification, check_file in sbml_tests:
            dct[f'test_{root}_{spec_name}'] = gen_test(specification, check_file)
        return type.__new__(mcs, names, bases, dct)

class TestFromSpecification(unittest.TestCase, metaclass=TestFromSpecificationMeta):
    TEST_ARGUMENTS = SimulatorArguments((0.0, 50.0), np.linspace(0, 50, 51))
    n = 1

    class TestResult(NamedTuple):
        results_df: pd.DataFrame
        check_df: pd.DataFrame
        z_scores_for_mean_by_species: pd.DataFrame

    def tearDown(self):
        # save results
        results_table = self.test_result.results_df
        z_ts = self.test_result.z_scores_for_mean_by_species
        results_table.to_csv(os.path.join(os.path.dirname(self.check_file), f'n={self.n}_simulation_results.csv'))
        z_ts.to_csv(os.path.join(os.path.dirname(self.check_file), f'n={self.n}_simulation_zscores.csv'))

    def _test_single(self):
        results = self.do_simulations()
        desired_species = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        all_species = [s.name for s in self.specification.model.species]
        targets = [all_species.index(s) for s in desired_species]
        aligned = self.align_results(results, self.check_data['time'], targets, desired_species)

        results_table, z_ts = self.z_score_for_mean(aligned, self.check_data, self.n)

        self.test_result = self.TestResult(results_table, self.check_data, z_ts)

        # assert something about zscores
        # TK

    def do_simulations(self):
        results = []
        simulator = self.specification.simulator
        forward_time = SIMULATORS[simulator]
        rng = np.random.default_rng()
        initial_condition = self.specification.model.make_initial_condition(self.specification.initial_condition)
        simulation_options = self.specification.simulation_options.copy()

        if simulator == 'hybrid':
            import hybrid
            partition_path = simulation_options.pop('partition')
            partition_scheme = hybrid.load_partition_scheme(partition_path)
            simulation_options['partition_function'] = partition_scheme.partition_function
        k = self.specification.model.get_k(parameters=self.specification.parameters, jit=True)
        for i in range(self.n):
            print(i)
            result = forward_time(initial_condition, self.TEST_ARGUMENTS.t_span, k, self.specification.model.stoichiometry(), self.specification.model.rate_involvement(), rng, discontinuities=self.TEST_ARGUMENTS.t_eval, **simulation_options)
            results.append(result)
        return results

    @staticmethod
    def align_results(results, time, target_indices, species_names):
        all_aligned = []
        for r in results:
            aligned = []
            t_history = r.t_history
            for t in time:
                idx = np.argmin(np.abs(t-t_history))
                aligned.append((r.t_history[idx], *[r.y_history[target_index,idx] for target_index in target_indices]))
            all_aligned.append(pd.DataFrame.from_records(aligned, columns=['time', *species_names]))

        indexed_results = []
        for r in all_aligned:
            r['time'] = np.round(r['time'], 5)
            r = r.set_index('time')
            indexed_results.append(r)

        return indexed_results

    @staticmethod
    def z_score_for_mean(aligned_results, check_data, n):
        target_species = set([c.split('-')[0] for c in check_data.columns if len(c.split('-')) > 1])

        df = pd.concat(aligned_results, axis=1)
        results_to_check = pd.concat([df.groupby(by=df.columns, axis=1).mean(), df.groupby(by=df.columns, axis=1).std()], axis=1)
        results_to_check.columns = [c + '-mean' if i < len(target_species) else c + '-sd' for i,c in enumerate(results_to_check.columns)]

        # https://github.com/sbmlteam/sbml-test-suite/blob/release/cases/stochastic/DSMTS-userguide-31v2.pdf
        z_ts = {}
        for species in target_species:
            z_t = (results_to_check[f'{species}-mean'] - check_data[f'{species}-mean'])/(check_data[f'{species}-sd']) * np.sqrt(n)
            z_ts[species] = z_t
        
        z_ts = pd.DataFrame(z_ts)

        return results_to_check, z_ts

if __name__ == '__main__':
    unittest.main()