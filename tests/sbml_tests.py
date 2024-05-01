import unittest
import os
import numpy as np
import pandas as pd
import pathlib
from dataclasses import dataclass
from typing import NamedTuple

from hybrid.simulate import SIMULATORS
from tests.discover_tests import discover_tests

sbml_tests = discover_tests(os.path.join(os.path.dirname(__file__), "sbml-tests/"), 'sbml-*')

# wherever we are, save test output to test_output folder
test_out = './test_output/'

@dataclass
class SimulatorArguments():
    t_span: tuple
    t_eval: tuple

class FilesystemTestMeta(type):
    test_collection = None
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

        for root, spec_name, specification, check_file in mcs.test_collection:
            test_name = f'{os.path.basename(os.path.normpath(root))}_{spec_name}'
            dct[f'test_{test_name}'] = gen_test(test_name, specification, check_file)
        return type.__new__(mcs, names, bases, dct)

class TestSBMLMeta(FilesystemTestMeta):
    test_collection = sbml_tests

class TestSpec(unittest.TestCase):
    TEST_ARGUMENTS = SimulatorArguments((0.0, 50.0), np.linspace(0, 50, 51))
    n = 10000

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
        for df in results:
            df.set_index('time', inplace=True)
        df = pd.concat(results, axis=1)
        all_results = pd.concat([df.groupby(by=df.columns, axis=1).mean(), df.groupby(by=df.columns, axis=1).std()], axis=1)

        check_targets = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        all_results.columns = [c + '-mean' if i < len(check_targets) else c + '-sd' for i,c in enumerate(all_results.columns)]

        z_ts = self.z_score_for_mean(all_results, check_targets, self.check_data, self.n)

        self.test_result = self.TestResult(all_results, self.check_data, z_ts)

        # assert something about zscores
        # TK

    def do_simulations(self, targets, desired_species):
        aligned_results = []
        rng = np.random.default_rng()
        initial_condition = self.specification.model.make_initial_condition(self.specification.initial_condition)
        simulator_config = self.specification.simulator_config.copy()
        simulator_class = simulator_config.pop('simulator')
        simulator_class = SIMULATORS[simulator_class]

        k = self.specification.model.get_k(parameters=self.specification.parameters, jit=True)
        print(simulator_config)
        simulator = simulator_class(k, self.specification.model.stoichiometry(), self.specification.model.kinetic_order(), **simulator_config)

        align_results = self.align_results_factory(self.check_data['time'], targets, desired_species)
        aligned_results = simulator.run_simulations(self.n, self.TEST_ARGUMENTS.t_span, initial_condition, rng=rng, t_eval=self.TEST_ARGUMENTS.t_eval, end_routine=align_results)

        return aligned_results

    def align_results_factory(self, check_data, targets, desired_species):
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

        return align_single_result

    @staticmethod
    def z_score_for_mean(all_results, target_species, check_data, n):
        # https://github.com/sbmlteam/sbml-test-suite/blob/release/cases/stochastic/DSMTS-userguide-31v2.pdf
        z_ts = {}
        for species in target_species:
            z_t = (all_results[f'{species}-mean'] - check_data[f'{species}-mean'])/(check_data[f'{species}-sd']) * np.sqrt(n)
            z_ts[species] = z_t

        z_ts = pd.DataFrame(z_ts)

        return z_ts

class TestSBML(TestSpec, metaclass=TestSBMLMeta):
    pass

if __name__ == '__main__':
    unittest.main()