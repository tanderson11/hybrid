import unittest
import os
import numpy as np
import pandas as pd
import pathlib
from typing import NamedTuple

from tests.discover_tests import discover_tests
from tests.filesystem_test import FilesystemTestMeta, TestSpec

sbml_tests = discover_tests(os.path.join(os.path.dirname(__file__), "sbml-tests/"), 'sbml-*')

# wherever we are, save test output to test_output folder
test_out = './test_output/'


class TestSBMLMeta(FilesystemTestMeta):
    test_collection = sbml_tests

class TestSBML(TestSpec, metaclass=TestSBMLMeta):
    def align_results_factory(self, time, targets, desired_species):
        def align_single_result(r):
            aligned = []
            t_history = r.t_history
            for t in time:
                idx = np.argmin(np.abs(t-t_history))
                aligned.append((r.t_history[idx], *[r.y_history[target_index,idx] for target_index in targets]))
            indexed_results = pd.DataFrame.from_records(aligned, columns=['time', *desired_species])
            indexed_results['time'] = np.round(indexed_results['time'], 5)
            indexed_results.set_index('time')

            return indexed_results

        return align_single_result

    class SBMLTestResult(NamedTuple):
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
        align_results = self.align_results_factory(self.check_data['time'], targets, desired_species)

        results = self.run_simulations(align_results)
        for df in results:
            df.set_index('time', inplace=True)
        df = pd.concat(results, axis=1)
        all_results = pd.concat([df.groupby(by=df.columns, axis=1).mean(), df.groupby(by=df.columns, axis=1).std()], axis=1)

        check_targets = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        all_results.columns = [c + '-mean' if i < len(check_targets) else c + '-sd' for i,c in enumerate(all_results.columns)]

        z_ts = self.z_score_for_mean(all_results, check_targets, self.check_data, self.n)

        self.test_result = self.SBMLTestResult(all_results, self.check_data, z_ts)

        # assert something about zscores
        # TK

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