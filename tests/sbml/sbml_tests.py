import unittest
import os
import numpy as np
import pandas as pd
import pathlib
from typing import NamedTuple

from tests.discover import discover_tests
from tests.filesystem_test import FilesystemTestMeta, MeanTest

sbml_tests = discover_tests(os.path.dirname(__file__), 'sbml-*', include_check=True, simulators_share_checks=True)

class SBMLCollection(FilesystemTestMeta):
    test_collection = sbml_tests

class ZScoreTest(MeanTest):
    t_eval = None
    do_yscores = True

    class ResultsAndCheck(NamedTuple):
        results_df: pd.DataFrame
        check_df: pd.DataFrame
        z_scores_for_mean_by_species: pd.DataFrame
        y_scores_for_std_by_species: pd.DataFrame = None

    def end_routine_factory(self, desired_species):
        super_end = super().end_routine
        def end_routine(result):
            df = super_end(result)
            df = df[list(desired_species)]
            return df
        return end_routine

    def tearDown(self):
        super().tearDown()
        # save results
        results_table = self.test_result.results_df
        z_ts = self.test_result.z_scores_for_mean_by_species
        out = os.path.join(self.test_out, self.test_name)
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        results_table.to_csv(os.path.join(out, f'n={self.n}_simulation_results.csv'))
        z_ts.to_csv(os.path.join(out, f'n={self.n}_simulation_zscores.csv'))

        if self.do_yscores:
            y_ts = self.test_result.y_scores_for_std_by_species
            y_ts.to_csv(os.path.join(out, f'n={self.n}_simulation_yscores.csv'))

    def _test_single(self):
        desired_species = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        end_routine = self.end_routine_factory(desired_species=desired_species)
        dfs = super()._test_single(end_routine=end_routine)
        #dfs = self.run_simulations(end_routine=end_routine)
        all_results = self.consolidate_data(dfs)
        check_targets = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])

        z_ts = self.z_score_for_mean(all_results, check_targets, self.check_data, self.n)
        y_ts = None
        if self.do_yscores:
            y_ts = self.y_score_for_std(all_results, check_targets, self.check_data, self.n)

        self.test_result = self.ResultsAndCheck(all_results, self.check_data, z_ts, y_ts)

    @staticmethod
    def z_score_for_mean(all_results, target_species, check_data, n):
        # https://github.com/sbmlteam/sbml-test-suite/blob/release/cases/stochastic/DSMTS-userguide-31v2.pdf
        z_ts = {}
        for species in target_species:
            z_t = (all_results[f'{species}-mean'] - check_data[f'{species}-mean'])/(check_data[f'{species}-sd']) * np.sqrt(n)
            z_ts[species] = z_t

        z_ts = pd.DataFrame(z_ts)

        return z_ts

    @staticmethod
    def y_score_for_std(all_results, target_species, check_data, n):
        y_ts = {}
        for species in target_species:
            y_t = (all_results[f'{species}-sd']**2/check_data[f'{species}-sd']**2 - 1) * np.sqrt(n/2)
            y_ts[species] = y_t

        y_ts = pd.DataFrame(y_ts)

        return y_ts

class TestSBML(ZScoreTest, metaclass=SBMLCollection):
    t_eval = np.linspace(0.0, 50.0, 51)

if __name__ == '__main__':
    unittest.main(failfast=True)